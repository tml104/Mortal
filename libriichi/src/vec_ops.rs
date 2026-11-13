use std::ops::AddAssign; // C++ 通过 operator+= 支持自定义类型的 +=；Rust 则通过实现 AddAssign trait 来支持它。

/*
trait AddAssign<Rhs = Self> {
    fn add_assign(&mut self, rhs: Rhs);
}
*/

/// Expecting some SIMD optimizations.
/// 右侧序列的元素，逐个加到左侧对应位置上，直接修改左侧序列。（也即主元素相加）
/// 如果两边长度不同，只处理较短序列覆盖的部分。
/// 例：[1, 2, 3] + [10, 20, 30] = [11, 22, 33]
#[inline]
pub(crate) fn vec_add_assign<L, R>(lhs: &mut [L], rhs: &[R]) // 在当前 crate 内可见。[L]：表示可变切片引用
where
    L: Copy + AddAssign<R>, // 约束：L 类型必须实现 Copy 和 AddAssign<R> trait，才能使用 += 操作符
    R: Copy,
{
    lhs.iter_mut() //创建一个迭代器: 逐个取得左侧元素的可变引用
        .zip(rhs) // 类似 Python 的 zip，把两边元素按位置配对。这里形成 (&mut L, &R) 的迭代器，其中LR为各个元素
        .for_each(|(l, &r)| *l += r); // 消费迭代器
}
