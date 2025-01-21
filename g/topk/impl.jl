# 原子类型定义，用于线程安全的操作
mutable struct Atomic{T}
    @atomic x::T
end

const Scalar = UInt64
const Bitwise = UInt64
const Index = Int32
const Count = Int32

const RADIX_BITS::Index = 2
const RADIX_SIZE::Index = 1 << RADIX_BITS
const RADIX_MASK::Bitwise = RADIX_SIZE - 1

@inbounds function convert(value::Int64)
    return Scalar(9223372036854775808) + value
end

@inbounds function convert(value::Float64)
    x::Scalar = reinterpret(Scalar, value)
    mask::Scalar = -((x >> 63)) | Scalar(0x8000000000000000)
    return (value == value) ? xor(x, mask) : Scalar(0xffffffffffffffff)
end

@inbounds function get_bit_field(
    data::Bitwise,
    location::Index,
    bits::Index,
)

    mask::Bitwise = (1 << bits) - 1
    return (data >> location) & mask
end

@inbounds function set_bit_field(
    data::Bitwise,
    insert::Bitwise,
    location::Index,
    bits::Index,
)

    mask::Bitwise = (1 << bits) - 1
    insert <<= location
    mask <<= location
    return (data & ~mask) | insert
end

@inbounds function count_radix_using_mask(
    data::AbstractVector{T},
    desired::Bitwise,
    mask::Bitwise,
    location::Index,
) where {T}

    counts::Array{Count} = zeros(Count, RADIX_SIZE)
    for value in data
        scalar::Scalar = convert(value)

        is_valid::Bool = (scalar & mask) == desired
        if !is_valid
            continue
        end

        radix::Index = get_bit_field(scalar, location, RADIX_BITS)
        counts[radix+1] += 1
    end

    return counts
end

@inbounds function find_pattern(
    data::AbstractVector{T},
    k::Index,
    desired::Bitwise,
) where {T}
    topk::Array{Index} = zeros(Index, k)
    topk_count::Index = 1

    for (index, value) in enumerate(data)
        scalar::Scalar = convert(value)

        if scalar >= desired
            topk[topk_count] = index
            topk_count += 1
        end
    end

    return topk
end

@inbounds function radix_select(
    data::AbstractVector{T},
    k::Index,
) where {T}

    desired::Bitwise = 0
    mask::Bitwise = 0

    k_remaining::Index = k
    for location::Index in
        (sizeof(Scalar)*8-RADIX_BITS):-RADIX_BITS:0

        counts::Array{Count} = count_radix_using_mask(
            data, desired, mask, location)

        for radix::Index in RADIX_SIZE-1:-1:0
            count::Count = counts[radix+1]

            if count == 1 && k_remaining == 1
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
end

@inbounds function topk(data::AbstractVector{T}, k) where {T}
    result::Array{Index} = radix_select(data, Index(k))
    return sort(result;
        lt=(x, y) ->
            data[x] > data[y] || (data[x] == data[y] && x < y))
end
