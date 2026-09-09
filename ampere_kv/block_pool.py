"""最小块编号池与单请求块表：只管理元数据，不分配张量或参与 Attention。"""


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


class BlockTable:
    """一个请求独占的有序块表；长度只表示登记的 Token 数，不证明 K/V 已写入。

    仅用于单线程。表中块必须由本表统一释放，外部不能直接向池归还这些编号。
    同一块池的各请求应使用相同块大小；当前池只管理编号，尚无实际存储布局。
    """

    def __init__(self, pool: BlockPool, block_size: int):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("每块 Token 数必须是正整数")
        self._pool = pool
        self._block_size = block_size
        self._block_ids: list[int] = []
        self._length = 0

    @property
    def length(self) -> int:
        """返回已登记的有效 Token 数，不包含最后一块的空闲位置。"""
        return self._length

    @property
    def block_ids(self) -> tuple[int, ...]:
        """按逻辑块顺序返回物理编号快照，避免调用方直接修改内部列表。"""
        return tuple(self._block_ids)

    def append_tokens(self, num_tokens: int) -> None:
        """登记尾部新增 Token；只申请缺少的块，不搬运或写入任何 K/V。"""
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("追加 Token 数必须是正整数")
        new_length = self._length + num_tokens
        # 向上取整：块大小为 4 时，长度 4 需要 1 块，长度 5 才需要 2 块。
        required = (new_length + self._block_size - 1) // self._block_size
        additional = required - len(self._block_ids)
        # 先检查全部需求，避免只分配一部分后才发现耗尽。
        # 依赖单线程且外部不破坏所有权；不承诺内存异常等故障下的事务回滚。
        if additional > self._pool.num_free_blocks:
            raise RuntimeError("空闲块不足，本次 Token 追加未执行")
        for _ in range(additional):
            self._block_ids.append(self._pool.allocate())
        self._length = new_length

    def release(self) -> None:
        """请求结束后归还全部块并重置长度；空表重复调用无操作，可重新使用。"""
        for block_id in self._block_ids:
            self._pool.free(block_id)
        # 清空持有的旧编号，防止下一次 release 再次释放已经归还的块。
        self._block_ids.clear()
        self._length = 0


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

    # 使用独立的小池检查单请求块表，不改变上面已验证的块池场景。
    pool = BlockPool(3)
    table = BlockTable(pool, block_size=4)
    assert table.length == 0 and table.block_ids == ()
    for count, expected_length, expected_blocks in ((3, 3, 1), (1, 4, 1), (2, 6, 2)):
        previous_ids = table.block_ids
        table.append_tokens(count)
        assert table.length == expected_length
        assert len(table.block_ids) == expected_blocks
        assert table.block_ids[:len(previous_ids)] == previous_ids
        assert pool.num_free_blocks == 3 - expected_blocks
        print(f"[PASS] 块表追加 {count} 个 Token：长度={table.length}，块数={expected_blocks}")

    # 长度 6 再加 7 得到 13，需要共 4 块；当前只剩 1 块，不能先占用它。
    # 同时检查非法追加的类型和值；拒绝后池与块表都必须完全不变。
    for count, error_type in ((7, RuntimeError), (0, ValueError), (-1, ValueError),
                              (True, ValueError), (1.0, ValueError)):
        before = (table.length, table.block_ids)
        expect_rejected(lambda: table.append_tokens(count), error_type)
        assert (table.length, table.block_ids) == before
    print("[PASS] 块不足及非法 Token 追加被拒绝，块表与块池状态不变")

    table.append_tokens(2)  # 长度到 8：填满已有两块，不申请新块。
    assert table.length == 8 and len(table.block_ids) == 2 and pool.num_free_blocks == 1
    table.append_tokens(1)  # 长度到 9：跨越边界，第三块只使用第一个位置。
    assert table.length == 9 and len(table.block_ids) == 3 and pool.num_free_blocks == 0
    assert len(set(table.block_ids)) == 3
    print("[PASS] 长度 8 不增块，长度 9 才申请第三块，块编号互不重复")

    table.release()
    assert table.length == 0 and table.block_ids == ()
    assert pool.num_free_blocks == 3 and not any(pool._allocated)
    before = (pool._free_blocks.copy(), pool._allocated.copy())
    table.release()  # 请求清理允许重复调用，但不能重复向池归还旧编号。
    assert (pool._free_blocks, pool._allocated) == before
    table.append_tokens(12)  # 清理后的空表可重新使用，全部三块都应可用。
    assert table.length == 12 and set(table.block_ids) == {0, 1, 2}
    assert pool.num_free_blocks == 0
    table.release()
    assert table.length == 0 and table.block_ids == () and pool.num_free_blocks == 3
    assert not any(pool._allocated)
    print("[PASS] 块表释放、重复清理及重新使用通过，最终全部块归还")
    print("[PASS] 单请求块表元数据自检通过；未验证真实 KV、位置映射、多请求或 GPU")


if __name__ == "__main__":
    main()
