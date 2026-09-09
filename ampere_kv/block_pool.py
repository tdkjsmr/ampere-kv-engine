"""最小块编号池：只管理分配与归还，不分配张量，也不参与 Attention。"""


class BlockPool:
    """单线程、独占块的编号管理器；不支持共享块、引用计数或请求归属检查。

    块编号只在本池内有效。调用方必须保管当前分配结果，释放后丢弃旧编号；
    编号复用后，本池无法区分合法调用与拿旧编号再次释放的错误调用。
    """

    def __init__(self, num_blocks: int):
        # bool 在 Python 中也是 int 的子类，这里只接受真正的内置整数。
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError("块数量必须是正整数")
        # 列表尾部作为栈顶：初次按 0、1、2……分配，归还的块优先复用。
        self._free_blocks = list(range(num_blocks - 1, -1, -1))
        # 下标就是块编号；标记用来拒绝未分配块的释放，避免重复加入空闲列表。
        self._allocated = [False] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        """返回剩余空闲块数；不是剩余 Token 数或显存字节数。"""
        return len(self._free_blocks)

    def allocate(self) -> int:
        """取出一个空闲块编号；耗尽时失败，不自动扩容或抢占其他块。"""
        if not self._free_blocks:
            raise RuntimeError("块池已耗尽，没有空闲块")
        block_id = self._free_blocks.pop()
        self._allocated[block_id] = True
        return block_id

    def free(self, block_id: int) -> None:
        """归还一个已分配编号；所有输入检查都在修改状态之前完成。"""
        if type(block_id) is not int or not 0 <= block_id < len(self._allocated):
            raise ValueError("块编号必须是池内有效整数")
        if not self._allocated[block_id]:
            raise ValueError("不能释放未分配或已经释放的块")
        # 这里只回收编号，不清零任何 K/V；未来读取仍必须遵守有效 Token 长度。
        self._free_blocks.append(block_id)
        self._allocated[block_id] = False


def main() -> None:
    """小型生命周期自检；只使用 Python，不加载模型或使用 GPU。"""
    pool = BlockPool(3)

    # 在当前场景逐项检查错误操作，并比较完整元数据，不能只检查空闲数量。
    def expect_rejected(operation, error_type):
        before = (pool._free_blocks.copy(), pool._allocated.copy())
        try:
            operation()
        except error_type:
            assert (pool._free_blocks, pool._allocated) == before
        else:
            raise AssertionError("非法操作没有被拒绝")

    # 块还没分配时不能归还；负数也不能被当作 Python 列表的倒数下标。
    for invalid_id in (0, -1, 3, True, 1.0):
        expect_rejected(lambda: pool.free(invalid_id), ValueError)
    print("[PASS] 未分配及非法块编号被拒绝，块池状态不变")

    blocks = [pool.allocate() for _ in range(3)]
    assert blocks == [0, 1, 2] and pool.num_free_blocks == 0
    assert all(pool._allocated)
    expect_rejected(pool.allocate, RuntimeError)
    print("[PASS] 三个块分配结果互不重复；耗尽后拒绝分配，状态不变")

    pool.free(blocks[1])
    assert pool.num_free_blocks == 1
    assert pool._allocated == [True, False, True]
    expect_rejected(lambda: pool.free(blocks[1]), ValueError)
    reused = pool.allocate()
    assert reused == blocks[1] and pool.num_free_blocks == 0
    print("[PASS] 已释放块拒绝重复归还，并可重新分配复用")

    # 归还当前持有的三个块，再完整分配一轮，检查没有块丢失或重复。
    for block_id in (blocks[0], reused, blocks[2]):
        pool.free(block_id)
    assert pool.num_free_blocks == 3 and not any(pool._allocated)
    blocks = [pool.allocate() for _ in range(3)]
    assert set(blocks) == {0, 1, 2} and pool.num_free_blocks == 0
    for block_id in blocks:
        pool.free(block_id)
    assert pool.num_free_blocks == 3 and not any(pool._allocated)
    print("[PASS] 全部归还后可再次完整分配，最终空闲块数恢复为 3")
    print("[PASS] 块编号池自检通过；未验证真实 KV 存储、页表、模型或 GPU")


if __name__ == "__main__":
    main()
