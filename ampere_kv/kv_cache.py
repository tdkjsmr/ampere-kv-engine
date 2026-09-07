"""单层、单请求的 BF16 连续 KV 存储；本文件不计算 Attention，也不加载模型。"""

import torch


class ContiguousKVCache:
    """固定容量、只允许尾部追加的推理缓存，不支持扩容或任意位置覆盖。"""

    def __init__(self, num_kv_heads: int, head_dim: int, capacity: int, device="cpu"):
        # 当前批大小固定为 1；头数、每头维度和容量由调用方明确给出。
        if num_kv_heads <= 0 or head_dim <= 0 or capacity <= 0:
            raise ValueError("KV 头数、每头维度和容量必须为正数")
        self.capacity = capacity
        self.length = 0
        shape = (1, num_kv_heads, capacity, head_dim)
        # empty 不初始化数据：只有已经写入的前缀才有效，未写入区域绝不能使用。
        self._key = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self._value = torch.empty(shape, dtype=torch.bfloat16, device=device)

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """复制外部算好的 K/V 到末尾；不记录梯度，不保留输入张量的引用。"""

        # 新数据必须是 [1, KV 头数, 本次 Token 数, 每头维度]，K/V 形状相同。
        # copy_ 本身允许广播和类型转换，这里主动拒绝，避免错误被悄悄掩盖。
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("新 K/V 必须是形状相同的四维张量")
        if key.shape[0] != 1 or key.shape[1] != self._key.shape[1] or key.shape[3] != self._key.shape[3]:
            raise ValueError("新 K/V 的批大小、KV 头数或每头维度与缓存不一致")
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise ValueError("新 K/V 必须使用 BF16")
        if key.device != self._key.device or value.device != self._value.device:
            raise ValueError("新 K/V 必须与缓存位于同一设备")
        tokens = key.shape[2]
        if tokens == 0:
            raise ValueError("每次追加至少需要一个 Token")
        end = self.length + tokens
        if end > self.capacity:
            raise ValueError(f"缓存容量不足：已用 {self.length}，本次追加 {tokens}，容量 {self.capacity}")

        # 左闭右开区间 [length, end)：只覆盖空闲位置，不拼接或重新分配缓存。
        # 所有可预检错误都在复制前拒绝；这里不提供 GPU 故障下的事务回滚保证。
        self._key[:, :, self.length:end, :].copy_(key)
        self._value[:, :, self.length:end, :].copy_(value)
        self.length = end

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回有效前缀视图；调用方应只读，修改返回值也会修改缓存。"""

        # 不返回未初始化的尾部，不额外复制历史数据。
        # 多头情况下，容量大于有效长度时，这个视图可能不是 contiguous 张量。
        return self._key[:, :, :self.length, :], self._value[:, :, :self.length, :]


def main() -> None:
    """显式运行本模块时，在 CPU 上做微型存储自检；不自动使用 GPU。"""

    cache = ContiguousKVCache(num_kv_heads=2, head_dim=4, capacity=6)
    # 地址只用于验证原地追加，没有在类的公开接口中增加调试功能。
    addresses = (cache._key.data_ptr(), cache._value.data_ptr())
    assert cache.length == 0
    assert all(tensor.shape == (1, 2, 0, 4) for tensor in cache.get())

    # 用小整数构造容易核对的 BF16 数据；K/V 不相同，便于发现混写错误。
    keys = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 4).to(torch.bfloat16)
    values = keys + 64
    start = 0
    for tokens in (3, 1, 2):
        # 依次覆盖：多 Token 写入、单 Token 追加、恰好写满容量。
        end = start + tokens
        cache.append(keys[:, :, start:end, :], values[:, :, start:end, :])
        actual_key, actual_value = cache.get()
        assert cache.length == end
        # 对照整个有效前缀：既检查新数据，也检查历史数据没有被覆盖。
        assert torch.equal(actual_key, keys[:, :, :end, :])
        assert torch.equal(actual_value, values[:, :, :end, :])
        assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
        print(f"[PASS] 追加 {tokens} 个 Token：有效长度={end}，内容正确，存储地址不变")
        start = end

    # 容量已满，再写一个 Token 必须失败；失败后长度、数据和地址均不变。
    try:
        cache.append(keys[:, :, :1, :], values[:, :, :1, :])
    except ValueError as error:
        print(f"[PASS] 越界写入被拒绝：{error}")
    else:
        raise AssertionError("[FAIL] 缓存已满却仍允许追加")
    actual_key, actual_value = cache.get()
    assert cache.length == 6
    assert torch.equal(actual_key, keys) and torch.equal(actual_value, values)
    assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses
    print("[PASS] 单层连续 KV 存储自检通过；此结果不代表模型或 GPU 验证通过")


if __name__ == "__main__":
    main()
