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
#include <istream>
#include <memory>
#include <span>
#include <sstream>
#include <vector>
#include "Utils.h"

namespace Search
{

/// @brief Metadata of index data field
struct IndexDataFieldMeta
{
    std::string name;
    size_t size;
    std::string checksum;
};

/**
 * @brief IndexDataReader is used to load index data from storage system.
*/
template <typename IStream>
class IndexDataReader
{
public:
    /// @brief Internal read buffer size.
    static const int BUFFER_SIZE = 4096;

    /// @brief Read a vector of field metadata.
    virtual std::vector<IndexDataFieldMeta> readFieldMeta() = 0;

    /**
     * @brief Get the input stream for field data.
     *
     * @param name name of the field
     * @return input stream of the field data, a null pointer if it doesn't exist
     */
    virtual std::shared_ptr<IStream>
    getFieldDataInputStream(const std::string & name) = 0;

    /// @brief Check whether a field exists.
    virtual bool hasFieldData(const std::string & name) = 0;

    /// @brief Read field's checksum, return whether the read is successful
    virtual bool readFieldChecksum(const std::string & name, size_t & checksum)
        = 0;

    virtual ~IndexDataReader() = default;

    /// @brief Reads the data of a given field into a string stream.
    /// @param name The name of the field to read data from.
    /// @param sstream The string stream where the field data will be stored.
    /// @return Returns true if the data read operation is successful, else returns false.
    bool readFieldData(const std::string & name, std::stringstream & sstream)
    {
        auto istream = getFieldDataInputStream(name);
        if (istream == nullptr)
            return false;

        std::vector<char> buffer(BUFFER_SIZE);
        while (!istream->eof())
        {
            istream->read(&buffer[0], BUFFER_SIZE);
            SI_THROW_IF_NOT_MSG(
                istream->eof() || !istream->fail(),
                ErrorCode::CANNOT_READ_FROM_ISTREAM,
                "IndexDataReader read failed");
            sstream.write(&buffer[0], istream->gcount());
        }
        return true;
    }
};

/**
 * @brief IndexDataWriter is used to save index data to storage system.
*/
template <typename OStream>
class IndexDataWriter
{
public:
    /// @brief Write a vector of field metadata.
    virtual bool writeFieldMeta(const std::vector<IndexDataFieldMeta> & fields)
        = 0;

    /// @brief Get the output stream for field data.
    virtual std::shared_ptr<OStream>
    getFieldDataOutputStream(const std::string & name) = 0;

    /// @brief Write data field's checksum to storage system.
    virtual void writeFieldChecksum(const std::string & name, size_t checksum)
        = 0;

    virtual ~IndexDataWriter() = default;
};

/**
 * @brief FileStore is used to manage files in local file system.
 *
 * It facilitates the loading and saving of index data into local cache files.
*/
template <typename IStream, typename OStream>
class FileStore
{
public:
    /**
     * @brief Get a file path based on field name and puts in FileStore.
     *
     * @param name name of the field.
     * @param temporary whether the file is temporary.
    */
    virtual std::string
    getFilePath(const std::string & name, bool temporary = false)
        = 0;

    /// @brief Remove a file from FileStore by path.
    virtual int remove(const std::string & path) = 0;

    /// @brief Remove a file from FileStore by field name.
    virtual int removeByName(const std::string & name) = 0;

    /// @brief Remove all temporary files in FileStore.
    virtual void removeTemporaryFiles() = 0;

    /**
     * @brief Mark a list of files as removed from the FileStore.
     *
     * These files are just marked as removed but not actually deleted from
     * the file system.
    */
    virtual void markRemoved(const std::vector<std::string> & files) = 0;

    /**
     * @brief Load a list of files from reader into local file system.
    */
    virtual void loadFiles(
        IndexDataReader<IStream> * reader,
        const std::vector<std::string> & names)
        = 0;

    /**
     * @brief Save local files managed by FileStore into storage system.
     *
     * Files marked as removed are skipped.
    */
    virtual void saveFiles(IndexDataWriter<OStream> * writer) = 0;

    /// @brief Write checksum file in the local file system.
    /// @param file_name local file name, will append checksum suffix for writing
    /// @param checksum checksum value written to the checksum file
    virtual void
    writeLocalChecksumFile(const std::string & file_name, size_t checksum)
        = 0;

    /// @brief Remove checksum file in the local file system.
    virtual bool removeLocalChecksumFile(const std::string & file_name) = 0;

    /// @brief Write local checksum file for a data field.
    inline void writeLocalChecksum(const std::string & name, size_t checksum)
    {
        writeLocalChecksumFile(this->getFilePath(name), checksum);
    }

    /// @brief Remove local checksum file for a data field.
    /// @param name name of the field
    inline void removeLocalChecksum(const std::string & name)
    {
        removeLocalChecksumFile(this->getFilePath(name));
    }

    virtual ~FileStore() = default;
};


/*
 * Generic IStream/OStream & Reader/Writer, used in DBMS
 */

/// @brief Base class of input stream for reading index data from storage.
class AbstractIStream
{
public:
    using seekdir = std::ios_base::seekdir;
    static const seekdir beg = std::ios_base::beg;

    /**
     * @brief Read a specific number of characters from the input stream.
     * @param s pointer to an array to store the data read
     * @param count maximum number of characters to read
     * @note Use `gcount()` to get the actual number of characters read.
    */
    virtual AbstractIStream & read(char * s, std::streamsize count) = 0;

    /// @brief Return whether the istream has been opened successfully.
    virtual bool is_open() const = 0;

    /// @brief Return true if an error has occurred on the associated stream.
    /// @note `eof()` also accounts as failure.
    virtual bool fail() const = 0;

    /// @brief Return whether an istream is at the end of the stream.
    virtual bool eof() const = 0;

    /// @brief Return number of characters read by last read operation.
    virtual std::streamsize gcount() const = 0;

    /// @brief Returns true if the istream has no errors and is ready for read
    /// operations. Specifically, returns !fail().
    virtual operator bool() const = 0;

    virtual ~AbstractIStream() = default;

    /// @brief Move cursor a specific location in the input stream.
    /// @note Be more compatible with ifstream, currently not used by
    /// faiss/diskann after refactoring
    virtual AbstractIStream & seekg(std::streampos offset, seekdir dir) = 0;
};

/// @brief Base class of output stream for writing index data into storage.
class AbstractOStream
{
public:
    using seekdir = std::ios_base::seekdir;
    static const seekdir beg = std::ios_base::beg;

    /// @brief Write a specific number of characters into the output stream.
    virtual AbstractOStream & write(const char * s, std::streamsize count) = 0;

    /// @brief Returns whether the last write opertion completed succesfully.
    virtual bool good() = 0;

    /// @brief Close the output stream.
    virtual void close() = 0;

    virtual ~AbstractOStream() = default;

    /// @brief Move cursor a specific location in the input stream.
    /// @note Be more compatible with ofstream, currently not used by
    /// faiss/diskann after refactoring
    virtual AbstractOStream & seekp(std::streampos offset, seekdir dir) = 0;
};

/**
 * @brief Write an array of data vectors to output stream.
 *
 * `StreamBinWriter` first writes (1) number of data points (2) vector dimension
 * into the output stream. Then it writes out the data in batches with `writeData()`.
 * Finally, call `finish()` to flush out the data and finish writing. Optionally,
 * it computes hash value of the data.
*/
template <typename T, typename OStream>
class StreamBinWriter
{
public:
    static const size_t IO_BLOCK_SIZE = (1UL << 20);

    StreamBinWriter(
        OStream & ostream_,
        size_t npts_,
        size_t ndims_,
        bool compute_hash_ = false) :
        ostream(ostream_),
        npts(npts_),
        ndims(ndims_),
        compute_hash(compute_hash_)
    {
        buffer.resize(IO_BLOCK_SIZE * 2);
        int32_t npts_i32 = static_cast<int32_t>(npts);
        int32_t ndims_i32 = static_cast<int32_t>(ndims);
        ostream.write(reinterpret_cast<char *>(&npts_i32), sizeof(int32_t));
        ostream.write(reinterpret_cast<char *>(&ndims_i32), sizeof(int32_t));
        bytes_written = sizeof(int32_t) * 2;
    }

    void writeData(const T * data, size_t len)
    {
        SI_THROW_IF_NOT(
            len * sizeof(T) <= IO_BLOCK_SIZE, ErrorCode::LOGICAL_ERROR);
        const char * data_ptr = reinterpret_cast<const char *>(data);
        std::copy(data_ptr, data_ptr + len * sizeof(T), buffer.data() + pos);
        pos += len * sizeof(T);
        if (pos >= IO_BLOCK_SIZE)
        {
            ostream.write(buffer.data(), IO_BLOCK_SIZE);
            SI_THROW_IF_NOT_MSG(
                ostream.good(),
                ErrorCode::CANNOT_WRITE_TO_OSTREAM,
                "StreamBinWriter write data failed");
            if (compute_hash)
                hash_value ^= buffer_hash(
                    std::string_view(buffer.data(), IO_BLOCK_SIZE));
            std::copy(
                buffer.data() + IO_BLOCK_SIZE,
                buffer.data() + pos,
                buffer.data());
            pos -= IO_BLOCK_SIZE;
            bytes_written += IO_BLOCK_SIZE;
        }
    }

    void finish()
    {
        if (pos)
        {
            ostream.write(buffer.data(), pos);
            SI_THROW_IF_NOT_MSG(
                ostream.good(),
                ErrorCode::CANNOT_WRITE_TO_OSTREAM,
                "StreamBinWriter finish writing data failed");
            if (compute_hash)
                hash_value ^= buffer_hash(std::string_view(buffer.data(), pos));
            bytes_written += pos;
            pos = 0;
        }
        SI_THROW_IF_NOT(
            bytes_written == sizeof(int32_t) * 2 + npts * ndims * sizeof(T),
            ErrorCode::LOGICAL_ERROR);
        SI_LOG_INFO(
            "save_bin: #pts={} #dims={} total_bytes={}, hash_value={}",
            npts,
            ndims,
            bytes_written,
            compute_hash ? hash_value : -1);
    }

    size_t bytesWritten() const { return bytes_written; }

    size_t hashValue() const { return hash_value; }

private:
    OStream & ostream;
    size_t npts;
    size_t ndims;
    bool compute_hash;
    std::vector<char> buffer;
    size_t pos{0};
    size_t bytes_written{0};

    size_t hash_value{0};
    std::hash<std::string_view> buffer_hash;
};

/**
 * @brief Read an array of data vectors from input stream.
 *
 * `StreamBinReader` first reads (1) number of data points (2) vector dimension
 * from the input stream. Then it reads the data in batches with `loadData()`.
*/
template <typename T, typename IStream>
class StreamBinReader
{
public:
    static const size_t IO_BLOCK_SIZE = (1UL << 20);

    StreamBinReader(IStream & istream_) : istream(istream_)
    {
        buffer.resize(IO_BLOCK_SIZE * 2);
        int32_t npts_i32 = 0, dim_i32 = 0;
        istream.read(reinterpret_cast<char *>(&npts_i32), sizeof(int32_t));
        SI_THROW_IF_NOT_MSG(
            !istream.fail() && istream.gcount() == sizeof(int32_t),
            ErrorCode::CANNOT_READ_FROM_ISTREAM,
            "StreamBinReader read npts failed");
        istream.read(reinterpret_cast<char *>(&dim_i32), sizeof(int32_t));
        SI_THROW_IF_NOT_MSG(
            !istream.fail() && istream.gcount() == sizeof(int32_t),
            ErrorCode::CANNOT_READ_FROM_ISTREAM,
            "StreamBinReader read dim failed");
        npts = static_cast<size_t>(npts_i32);
        ndims = static_cast<size_t>(dim_i32);
    }

    size_t nptsValue() const { return npts; }

    size_t ndimsValue() const { return ndims; }

    size_t loadData(T * data, size_t len)
    {
        size_t data_size = len * sizeof(T);
        SI_THROW_IF_NOT(data_size <= IO_BLOCK_SIZE, ErrorCode::LOGICAL_ERROR);
        if (buffer_st + data_size >= buffer_end)
        {
            // need to read more data
            if (buffer_end > IO_BLOCK_SIZE)
            {
                // shift the buffer starting point to prepare for read
                std::copy(
                    buffer.data() + buffer_st,
                    buffer.data() + buffer_end,
                    buffer.data());
                buffer_end -= buffer_st;
                buffer_st = 0;
            }
            istream.read(buffer.data() + buffer_end, IO_BLOCK_SIZE);
            SI_THROW_IF_NOT_MSG(
                istream.eof() || !istream.fail(),
                ErrorCode::CANNOT_READ_FROM_ISTREAM,
                "StreamBinReader read data failed");
            buffer_end += istream.gcount();
        }
        // copy from buffer to output
        size_t read_end = std::min(buffer_st + data_size, buffer_end);
        SI_THROW_IF_NOT(
            (read_end - buffer_st) % sizeof(T) == 0, ErrorCode::LOGICAL_ERROR);
        size_t num_read = (read_end - buffer_st) / sizeof(T);
        std::copy(
            buffer.data() + buffer_st,
            buffer.data() + read_end,
            reinterpret_cast<char *>(data));
        buffer_st = read_end;
        return num_read;
    }

private:
    IStream & istream;
    std::vector<char> buffer;
    size_t buffer_st{0};
    size_t buffer_end{0};
    size_t npts;
    size_t ndims;
};

/**
 * @brief Save an array of data vectors to output stream.
 * @return Total number of bytes written.
 *
 * It uses `StreamBinWriter` internally.
*/
template <typename T, typename OStream>
inline uint64_t save_bin_with_writer(
    OStream & ostream,
    const T * data,
    size_t npts,
    size_t ndims,
    size_t * checksum = nullptr)
{
    StreamBinWriter<T, OStream> stream_writer(ostream, npts, ndims);
    size_t batch_size = stream_writer.IO_BLOCK_SIZE / sizeof(T);
    for (size_t st = 0; st < npts * ndims; st += batch_size)
    {
        size_t len = std::min(batch_size, npts * ndims - st);
        stream_writer.writeData(data + st, len);
        SIConfiguration::currentThreadCheckAndAbort();
    }
    stream_writer.finish();
    ostream.close();
    if (checksum)
        *checksum = stream_writer.hashValue();
    return stream_writer.bytesWritten();
}

/**
 * @brief Load an array of data vectors from input stream.
 *
 * It uses `StreamBinReader` internally.
*/
template <typename T, typename IStream>
inline void load_bin_from_reader_into_data(
    StreamBinReader<T, IStream> & stream_reader,
    T * data,
    size_t npts,
    size_t ndims)
{
    size_t batch_size = stream_reader.IO_BLOCK_SIZE / sizeof(T);
    for (size_t st = 0; st < npts * ndims; st += batch_size)
    {
        size_t len = std::min(batch_size, npts * ndims - st);
        stream_reader.loadData(data + st, len);
        SIConfiguration::currentThreadCheckAndAbort();
    }
}

template <typename T, typename IStream>
inline void load_bin_from_reader(
    IStream & istream, std::vector<T> & data, size_t & npts, size_t & ndims)
{
    StreamBinReader<T, IStream> stream_reader(istream);
    npts = stream_reader.nptsValue();
    ndims = stream_reader.ndimsValue();

    SI_LOG_INFO("load_bin: #pts={} #dims={}", npts, ndims);
    checkAvailableMemory(npts * ndims * sizeof(T));
    data.resize(npts * ndims);

    load_bin_from_reader_into_data(stream_reader, data.data(), npts, ndims);
}

template <typename T, typename IStream>
inline void load_bin_from_reader(
    IStream & istream,
    std::function<std::span<T>(size_t, size_t)> get_data,
    size_t & npts,
    size_t & ndims)
{
    StreamBinReader<T, IStream> stream_reader(istream);
    npts = stream_reader.nptsValue();
    ndims = stream_reader.ndimsValue();
    SI_LOG_INFO("load_bin: #pts={} #dims={}", npts, ndims);

    std::span<T> data = get_data(npts, ndims);
    load_bin_from_reader_into_data(stream_reader, data.data(), npts, ndims);
}

} // namespace Search
