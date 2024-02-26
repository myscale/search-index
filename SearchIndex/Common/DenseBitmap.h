/*
 * Copyright © 2024 MOQI SINGAPORE PTE. LTD.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 */

#pragma once
#include <bit>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "Utils.h"

namespace Search
{

class DenseBitmap;
using DenseBitmapPtr = std::shared_ptr<DenseBitmap>;

/**
 * @brief A dense bitmap for lightweight delete and filter search.
*/
class DenseBitmap
{
public:
    /// @brief Align the byte size of DenseBitmap to be multiple of 8.
    static const int BYTE_ALIGNMENT = 8;

    DenseBitmap() = default;

    /// @brief Construct a bitmap with given size and value.
    explicit DenseBitmap(size_t size_, bool value = false) : size(size_)
    {
        bitmap = new uint8_t[byte_size()];
        memset(bitmap, value ? 255 : 0, byte_size());
    }

    DenseBitmap(const DenseBitmap & other)
    {
        size = other.size;
        bitmap = new uint8_t[byte_size()];
        memcpy(bitmap, other.bitmap, byte_size());
    }

    /// @brief Return number elements in the bitmap
    size_t get_size() const { return size; }

    /**
     * @brief Return the byte size of the bitmap.
     * @note The byte size is aligned to be multiple of BYTE_ALIGNMENT.
     */
    inline size_t byte_size() const
    {
        return DIV_ROUND_UP(size, 8 * BYTE_ALIGNMENT) * BYTE_ALIGNMENT;
    }

    /// @brief Return the raw bitmap data.
    uint8_t * get_bitmap() { return bitmap; }

    /// @brief Check whether `id` is in the bitmap.
    /// @note This function will throw if `id` is out of range.
    inline bool is_member(size_t id) const
    {
        SI_THROW_IF_NOT(id < size, ErrorCode::LOGICAL_ERROR);
        return unsafe_test(id);
    }

    /// @brief Check whether `id` is in the bitmap without boundary check.
    inline bool unsafe_test(size_t id) const
    {
        return (bitmap[id >> 3] & (0x1 << (id & 0x7)));
    }

    /**
     * @brief Approximately count the number of bits set to 1 around `id`.
     * 
     * This function is used to estimate the cardinality of the bitmap for 
     * filter search parameter auto-tuning.
     */
    inline std::pair<size_t, size_t> unsafe_approx_count_member(size_t id) const
    {
        SI_THROW_IF_NOT_FMT(
            id < get_size(),
            ErrorCode::LOGICAL_ERROR,
            "id must be smaller than bitmap size: %lu vs. %lu",
            id,
            get_size());
        size_t idx = (id >> 3) / BYTE_ALIGNMENT * BYTE_ALIGNMENT;
        SI_THROW_IF_NOT_FMT(
            idx + 7 < byte_size(),
            ErrorCode::LOGICAL_ERROR,
            "idx+7 exceededs byte_size: %lu vs. %lu",
            idx + 7,
            byte_size());
        size_t count
            = std::popcount(*reinterpret_cast<uint64_t *>(&bitmap[idx]));
        return std::make_pair(64, count);
    }

    /// @brief set bit value at `id` to 1 in the bitmap
    /// @note this function will throw if `id` is out of range
    inline void set(size_t id)
    {
        SI_THROW_IF_NOT(id < size, ErrorCode::LOGICAL_ERROR);
        bitmap[id >> 3] |= (0x1 << (id & 0x7));
    }

    /// @brief set bit value at `id` to 0 in the bitmap
    /// @note this function will throw if `id` is out of range
    inline void unset(size_t id)
    {
        SI_THROW_IF_NOT(id < size, ErrorCode::LOGICAL_ERROR);
        bitmap[id >> 3] &= ~(0x1 << (id & 0x7));
    }

    /// @brief Check whether all bits are set to 1 in the bitmap.
    bool all() const
    {
        for (size_t i = 0; i < byte_size(); ++i)
        {
            if (bitmap[i] != 255)
                return false;
        }
        return true;
    }

    /// @brief Check whether any bit is set to 1 in the bitmap.
    bool any() const
    {
        for (size_t i = 0; i < byte_size(); ++i)
        {
            if (bitmap[i])
                return true;
        }
        return false;
    }

    /// @brief Count the number of bits set to 1 in the bitmap.
    size_t count() const
    {
        size_t count = 0;
        for (size_t i = 0; i < byte_size(); ++i)
        {
            count += __builtin_popcount(bitmap[i]);
        }
        return count;
    }

    /// @brief Free the memory allocated by the bitmap.
    ~DenseBitmap()
    {
        // delete null pointer has no effect
        delete[] bitmap;
    }

    /// @brief Return the raw bitmap data.
    uint8_t * data() { return bitmap; }

    /// @brief Return indices of all the bits set to 1 as a vector.
    std::vector<size_t> to_vector() const
    {
        std::vector<size_t> result;
        result.reserve(count());
        for (size_t i = 0; i < size; ++i)
        {
            if (unsafe_test(i))
                result.push_back(i);
        }
        return result;
    }

private:
    /// @brief The raw bitmap data.
    uint8_t * bitmap;

    /// @brief Number of elements in the bitmap.
    size_t size;

    friend DenseBitmapPtr
    intersectDenseBitmaps(DenseBitmapPtr left, DenseBitmapPtr right);
};

/// @brief Intersect elements of two dense bitmaps.
inline DenseBitmapPtr
intersectDenseBitmaps(DenseBitmapPtr left, DenseBitmapPtr right)
{
    if (left == nullptr)
        return right;
    else if (right == nullptr)
        return left;
    SI_THROW_IF_NOT_FMT(
        left->get_size() == right->get_size(),
        ErrorCode::LOGICAL_ERROR,
        "left size %zu != right size %zu",
        left->get_size(),
        right->get_size());
    DenseBitmapPtr after_merge
        = std::make_shared<DenseBitmap>(left->get_size());
    uint8_t * bits = after_merge->bitmap;
    uint8_t * left_bits = left->bitmap;
    uint8_t * right_bits = right->bitmap;
    for (size_t i = 0; i < left->byte_size(); ++i)
    {
        bits[i] = left_bits[i] & right_bits[i];
    }
    return after_merge;
}
}
