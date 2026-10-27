#pragma once

// =============================================================================
// naga/cospan/wrappers.hpp — the device-side data handles (r02 section 4)
// =============================================================================
//
// Input, Output and Initializer are pure device payload: {shape, stride, data},
// trivially copyable, no identity. Their fields are private and written only by
// detail::WrapperAccess (friendship): user code never sees a pointer. Indexing is by
// sycl::id<Rank> or by a flat row-major index over shape(); stride() is the element
// step per dimension in the OWNING allocation. LocalArray declares an array of a
// group-range stage's local memory and LocalView is the stage's view of it.

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "naga/cospan/dtype.hpp"

namespace naga::cospan {

struct CustomOp;  // the one place a raw device pointer is obtainable (r02 section 4.1)

namespace detail {

struct WrapperAccess;
template <class T, int Rank>
struct RaggedAccess;
struct ItemFixup;

/// A range of rank R taken from the first R entries of a 1-padded triple.
template <int R>
constexpr sycl::range<R> make_range(const std::array<std::size_t, 3>& a) {
    if constexpr (R == 1)
        return sycl::range<1>{a[0]};
    else if constexpr (R == 2)
        return sycl::range<2>{a[0], a[1]};
    else
        return sycl::range<3>{a[0], a[1], a[2]};
}

/// A range of rank R with every extent `v`.
template <int R>
constexpr sycl::range<R> filled_range(std::size_t v) {
    return make_range<R>({v, v, v});
}

template <int R>
constexpr std::size_t range_size(const sycl::range<R>& r) {
    std::size_t n = 1;
    for (int d = 0; d < R; ++d) n *= r[d];
    return n;
}

/// Element offset of a multi-index under `stride`.
template <int R>
constexpr std::size_t offset_of(const sycl::id<R>& idx, const sycl::range<R>& stride) {
    std::size_t o = 0;
    for (int d = 0; d < R; ++d) o += idx[d] * stride[d];
    return o;
}

/// Element offset of a flat row-major index over `shape`, under `stride`.
template <int R>
constexpr std::size_t offset_of_flat(std::size_t flat, const sycl::range<R>& shape,
                                     const sycl::range<R>& stride) {
    std::size_t o = 0;
    for (int d = R - 1; d >= 0; --d) {
        o += (flat % shape[d]) * stride[d];
        flat /= shape[d];
    }
    return o;
}

}  // namespace detail

/// Input<DT, Rank> — read-only view. A kernel cannot write through an Input:
/// all access is const-reference. Atomic dtypes are ill-formed on Input.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct Input {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    const value_type& operator[](sycl::id<Rank> idx) const {
        return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    const value_type& operator[](std::size_t flat) const {
        return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }

    /// Adjoint views only: atomically accumulates into the element (casts away const). Derivative kernels accumulate, never assign.
    void add(sycl::id<Rank> idx, value_type v) const {
        value_type* p = const_cast<value_type*>(data_ + detail::offset_of<Rank>(idx, stride_));
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>{*p}.fetch_add(v);
    }
    void add(std::size_t flat, value_type v) const {
        value_type* p = const_cast<value_type*>(data_ + detail::offset_of_flat<Rank>(flat, shape_, stride_));
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>{*p}.fetch_add(v);
    }

    sycl::range<Rank> shape() const { return shape_; }
    sycl::range<Rank> stride() const { return stride_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    sycl::range<Rank> stride_ = detail::filled_range<Rank>(0);
    const value_type* data_   = nullptr;
    friend struct detail::WrapperAccess;
    friend struct CustomOp;
};

/// Output<DT, Rank> — read-write view with exactly ONE access discipline per
/// instantiation: bare mutable references for plain dtypes, sycl::atomic_ref
/// (relaxed, device scope) for atomic dtypes. No bare reference is obtainable
/// from an atomic Output.
template <DType DT, int Rank>
struct Output {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    using atomic_type =
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    value_type& operator[](std::size_t flat) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }

    atomic_type operator[](sycl::id<Rank> idx) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of<Rank>(idx, stride_)]};
    }
    atomic_type operator[](std::size_t flat) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)]};
    }

    /// The value of the element (an atomic dtype loads atomically).
    value_type get(sycl::id<Rank> idx) const {
        if constexpr (is_atomic_v<DT>) return atomic_type{data_[detail::offset_of<Rank>(idx, stride_)]}.load();
        else return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    value_type get(std::size_t flat) const {
        if constexpr (is_atomic_v<DT>) return atomic_type{data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)]}.load();
        else return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }
    /// Atomically accumulates into the element (derivative kernels accumulate, never assign).
    void add(sycl::id<Rank> idx, value_type v) const
        requires(!is_atomic_v<DT>)
    {
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>{
            data_[detail::offset_of<Rank>(idx, stride_)]}.fetch_add(v);
    }
    void add(std::size_t flat, value_type v) const
        requires(!is_atomic_v<DT>)
    {
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>{
            data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)]}.fetch_add(v);
    }

    sycl::range<Rank> shape() const { return shape_; }
    sycl::range<Rank> stride() const { return stride_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    sycl::range<Rank> stride_ = detail::filled_range<Rank>(0);
    value_type*       data_   = nullptr;
    friend struct detail::WrapperAccess;
    friend struct CustomOp;
};

/// Initializer<DT, Rank> — the writable twin of an Output for a custom initializer (cs::init::custom): the same
/// accessor discipline as Output, over the variable's data on the device the initializer runs on (r02 section 2.5).
template <DType DT, int Rank>
struct Initializer {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    using atomic_type =
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    value_type& operator[](std::size_t flat) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }

    atomic_type operator[](sycl::id<Rank> idx) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of<Rank>(idx, stride_)]};
    }
    atomic_type operator[](std::size_t flat) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)]};
    }

    sycl::range<Rank> shape() const { return shape_; }
    sycl::range<Rank> stride() const { return stride_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    sycl::range<Rank> stride_ = detail::filled_range<Rank>(0);
    value_type*       data_   = nullptr;
    friend struct detail::WrapperAccess;
    friend struct CustomOp;
};

/// LocalView<DT, Rank> — a stage's view of one LocalArray in group-local memory: a pointer and the array's shape, contiguous
/// row-major. Trivially copyable; made by LocalMem::get.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct LocalView {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;

    LocalView(value_type* data, sycl::range<Rank> shape) : data_{data}, shape_{shape} {}

    value_type& operator[](sycl::id<Rank> idx) const {
        std::size_t flat = 0;
        for (int d = 0; d < Rank; ++d) flat = flat * shape_[d] + idx[d];
        return data_[flat];
    }
    value_type&       operator[](std::size_t flat) const { return data_[flat]; }
    sycl::range<Rank> shape() const { return shape_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    value_type*       data_;
    sycl::range<Rank> shape_;
};

/// LocalArray<DT, Rank> — the declaration of one array of a group-range stage's local memory (a template argument of
/// LocalMem): non-atomic dtypes only. It holds no data; LocalMem::get hands out the LocalView of it.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct LocalArray {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    using view_type  = LocalView<DT, Rank>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;
};

// ---- ragged wrappers -------------------------------------------------------------

/// A dense row-major view of one block of a ragged port: rank 1 (ragged is rank 2 only: the port's rank - 1), shape per
/// block. `T` is const-qualified for an input. Trivially copyable; made by RaggedInput/RaggedOutput::block.
template <class T, int R>
struct Block {
    static_assert(R == 1, "cospan: a ragged block has rank 1 (ragged is rank 2 only)");
    T& operator[](sycl::id<R> idx) const {
        std::size_t flat = 0;
        for (int d = 0; d < R; ++d) flat = flat * shape_[d] + idx[d];
        return data_[flat];
    }
    T&                operator[](std::size_t flat) const { return data_[flat]; }
    sycl::range<R>    shape() const { return shape_; }
    std::size_t       size() const { return detail::range_size<R>(shape_); }

private:
    template <class, int>
    friend struct detail::RaggedAccess;
    sycl::range<R> shape_ = detail::filled_range<R>(0);
    T*             data_  = nullptr;
};

/// The block of an ATOMIC ragged output: the shape of a Block, with operator[] returning a relaxed, device-scope
/// sycl::atomic_ref (as a dense atomic Output does); no bare reference is obtainable.
template <class T, int R>
struct AtomicBlock {
    using atomic_type = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    AtomicBlock() = default;
    explicit AtomicBlock(Block<T, R> b) : b_{b} {}
    atomic_type    operator[](sycl::id<R> idx) const { return atomic_type{b_[idx]}; }
    atomic_type    operator[](std::size_t flat) const { return atomic_type{b_[flat]}; }
    sycl::range<R> shape() const { return b_.shape(); }
    std::size_t    size() const { return b_.size(); }

private:
    Block<T, R> b_{};
};

namespace detail {

/// Shared block lookup of RaggedInput / RaggedOutput. Per dim d >= 1 either a scalar extent (ext[d-1])
/// or a per-index array (per[d-1] non-null). offsets == nullptr: a dense port, block i is row i.
template <class T, int Rank>
struct RaggedAccess {
    static Block<T, Rank - 1> block(T* data, const std::uint32_t* offsets, const std::array<std::size_t, 2>& ext,
                                    const std::array<const std::uint32_t*, 2>& per, std::size_t i) {
        Block<T, Rank - 1> b;
        std::size_t        size = 1;
        for (int d = 1; d < Rank; ++d) {
            const std::size_t e = per[d - 1] != nullptr ? static_cast<std::size_t>(per[d - 1][i]) : ext[d - 1];
            b.shape_[d - 1]     = e;
            size *= e;
        }
        // Relative to offsets[0]: under a leading range `offsets` points at the root's offsets[a]
        // and `data` at the root's block a. An unshifted port has offsets[0] == 0.
        b.data_ = data + (offsets != nullptr ? static_cast<std::size_t>(offsets[i] - offsets[0]) : i * size);
        return b;
    }
};

}  // namespace detail

/// RaggedInput<DT, Rank> — read-only ragged port, Rank 2 only: N blocks, block i of shape (extent_1(i)).
/// Also accepts a DENSE port (block i = row i). Non-atomic only.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct RaggedInput {
    static_assert(Rank == 2, "cospan: a ragged port is rank 2 only");
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype  = DT;
    static constexpr int   rank   = Rank;
    static constexpr bool  ragged = true;

    std::size_t size() const { return n_; }
    Block<const value_type, Rank - 1> block(std::size_t i) const {
        return detail::RaggedAccess<const value_type, Rank>::block(data_, offsets_, ext_, per_, i);
    }

private:
    std::size_t                         n_       = 0;
    const value_type*                   data_    = nullptr;
    const std::uint32_t*                offsets_ = nullptr;
    std::array<std::size_t, 2>          ext_{1, 1};
    std::array<const std::uint32_t*, 2> per_{nullptr, nullptr};
    friend struct detail::WrapperAccess;
};

/// RaggedOutput<DT, Rank> — the writable twin. With an atomic DT, block(i) is an AtomicBlock (operator[] gives a
/// sycl::atomic_ref, relaxed, device scope), mirroring a dense atomic Output.
template <DType DT, int Rank>
struct RaggedOutput {
    static_assert(Rank == 2, "cospan: a ragged port is rank 2 only");
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype  = DT;
    static constexpr int   rank   = Rank;
    static constexpr bool  ragged = true;

    std::size_t size() const { return n_; }
    Block<value_type, Rank - 1> block(std::size_t i) const
        requires(!is_atomic_v<DT>)
    {
        return detail::RaggedAccess<value_type, Rank>::block(data_, offsets_, ext_, per_, i);
    }
    AtomicBlock<value_type, Rank - 1> block(std::size_t i) const
        requires(is_atomic_v<DT>)
    {
        return AtomicBlock<value_type, Rank - 1>{detail::RaggedAccess<value_type, Rank>::block(data_, offsets_, ext_, per_, i)};
    }

private:
    std::size_t                         n_       = 0;
    value_type*                         data_    = nullptr;
    const std::uint32_t*                offsets_ = nullptr;
    std::array<std::size_t, 2>          ext_{1, 1};
    std::array<const std::uint32_t*, 2> per_{nullptr, nullptr};
    friend struct detail::WrapperAccess;
};

static_assert(std::is_trivially_copyable_v<RaggedInput<DType::f32, 2>>);
static_assert(std::is_trivially_copyable_v<RaggedOutput<DType::f32, 2>>);
static_assert(std::is_trivially_copyable_v<Block<float, 1>>);
static_assert(std::is_trivially_copyable_v<RaggedOutput<DType::atomic_f32, 2>>);
static_assert(std::is_trivially_copyable_v<AtomicBlock<float, 1>>);

static_assert(std::is_trivially_copyable_v<Input<DType::f32, 2>>);
static_assert(std::is_trivially_copyable_v<Output<DType::atomic_f32, 1>>);
static_assert(std::is_trivially_copyable_v<Initializer<DType::f32, 1>>);
static_assert(std::is_trivially_copyable_v<LocalView<DType::f32, 2>>);

}  // namespace naga::cospan