using Base.Threads

# 原子类型定义，用于线程安全的操作
mutable struct Atomic{T}
    @atomic x::T
end

const Scalar = UInt64
const Bitwise = UInt64
const Index = Int32
const Count = Int32

const RADIX_BITS::Index = 8
const RADIX_SIZE::Index = 1 << RADIX_BITS
const RADIX_MASK::Bitwise = RADIX_SIZE - 1

const NUM_THREADS::Index = 8

@inbounds function convert(value::Int64)::Scalar
    return Scalar(9223372036854775808) + value
end

@inbounds function convert(value::Float64)::Scalar
    x::Scalar = reinterpret(Scalar, value)
    mask::Scalar = -((x >> 63)) | Scalar(0x8000000000000000)
    return (value == value) ? xor(x, mask) : Scalar(0xffffffffffffffff)
end

@inbounds function get_bit_field(
    data::Bitwise,
    location::Index,
    bits::Index,
)::Bitwise

    mask::Bitwise = (1 << bits) - 1
    return (data >> location) & mask
end

@inbounds function set_bit_field(
    data::Bitwise,
    insert::Bitwise,
    location::Index,
    bits::Index,
)::Bitwise

    mask::Bitwise = (1 << bits) - 1
    insert <<= location
    mask <<= location
    return (data & ~mask) | insert
end

@inbounds function count_radix_using_mask!(
    data::AbstractVector{T},
    desired::Bitwise,
    mask::Bitwise,
    location::Index,
    counts::Vector{Count},
) where {T}

    counts_local::Vector{Count} = zeros(Count, RADIX_SIZE * NUM_THREADS)
    @threads for value in data
        scalar::Scalar = convert(value)

        is_valid::Bool = (scalar & mask) == desired
        if !is_valid
            continue
        end

        radix::Index = get_bit_field(scalar, location, RADIX_BITS)
        thread_index::Index = threadid()
        counts_local[radix+(thread_index-1)*RADIX_SIZE+1] += one(Count)
    end

    counts .= zero(Count)
    @simd for offset in 0:NUM_THREADS-1
        @simd for i in eachindex(counts)
            counts[i] += counts_local[i+offset*RADIX_SIZE]
        end
    end
end

@inbounds function find_pattern(
    data::AbstractVector{T},
    k::Index,
    desired::Bitwise,
)::Vector{Index} where {T}

    topk::Vector{Index} = Vector{Index}(undef, k)
    topk_count::Atomic{Index} = Atomic{Index}(one(Index))

    @threads for i in eachindex(data)
        scalar::Scalar = convert(data[i])

        if scalar >= desired
            topk_index::Index = (
                @atomic topk_count.x += one(Index)
            ) - 1

            topk[topk_index] = i
        end
    end

    return topk
end

@inbounds function radix_select(
    data::AbstractVector{T},
    k::Index,
)::Vector{Index} where {T}

    desired::Bitwise = zero(Bitwise)
    mask::Bitwise = zero(Bitwise)

    k_remaining::Index = k
    counts::Vector{Count} = Vector{Count}(undef, RADIX_SIZE)
    for location::Index in
        (sizeof(Scalar)*8-RADIX_BITS):-RADIX_BITS:0

        count_radix_using_mask!(
            data, desired, mask, location, counts)

        for radix::Index in RADIX_SIZE-1:-1:0
            count::Count = counts[radix+1]

            if count == one(Count) && k_remaining == one(Index)
                desired = set_bit_field(desired, Bitwise(radix), location, RADIX_BITS)
                mask = set_bit_field(mask, RADIX_MASK, location, RADIX_BITS)
                return find_pattern(data, k, desired)
            end

            if count >= k_remaining
                desired = set_bit_field(desired, Bitwise(radix), location, RADIX_BITS)
                mask = set_bit_field(mask, RADIX_MASK, location, RADIX_BITS)
                break
            end

            k_remaining -= count
        end
    end

    return Index[]
end

@inbounds function topk(data::AbstractVector{T}, k) where {T}
    result::Vector{Index} = radix_select(data, Index(k))
    return sort(result;
        lt=(x, y) ->
            data[x] > data[y] || (data[x] == data[y] && x < y))
end
