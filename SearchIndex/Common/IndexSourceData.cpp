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

#include "IndexSourceData.h"

namespace Search
{

template <>
size_t DataSet<float>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(float);
}

template <>
size_t DataSet<double>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(double);
}

template <>
size_t DataSet<long>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(long);
}

template <>
size_t DataSet<int>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(int);
}

template <>
size_t DataSet<short>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(short);
}

template <>
size_t DataSet<signed char>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(signed char);
}

template <>
size_t DataSet<unsigned char>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(unsigned char);
}

template <>
size_t DataSet<unsigned short>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(unsigned short);
}

template <>
size_t DataSet<unsigned int>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(unsigned int);
}

template <>
size_t DataSet<bool>::singleVectorSizeInByte() const
{
    return (dimension() + 7) / 8;
}
}
