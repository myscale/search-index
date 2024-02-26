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
#include <concepts>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <SearchIndex/Common/IndexDataIO.h>
#include <SearchIndex/SearchIndexCommon.h>


namespace Search
{

// index filename patterns
static const std::string INDEX_META_FILENAME = "metadata.txt";
static const std::string INDEX_SUFFIX = ".vidx3";

inline std::string INDEX_DATA_FILED_FILENAME_FMT(const std::string & name)
{
    return name + INDEX_SUFFIX;
}

inline std::string INDEX_CHECKSUM_FILED_FILENAME_FMT(const std::string & name)
{
    return name + ".checksum" + INDEX_SUFFIX;
}

/// @brief Load index data from local files.
template <typename IS>
class IndexDataFileReader final : public IndexDataReader<IS>
{
public:
    static const size_t META_BUFFER_SIZE = 4096;

    using IStreamConstructor = std::function<std::shared_ptr<IS>(
        const std::string & name, std::ios::openmode)>;

    IndexDataFileReader(
        const std::string & path_prefix_,
        IStreamConstructor istream_constructor_) :
        path_prefix(path_prefix_),
        istream_constructor(std::move(istream_constructor_))
    {
    }

    /// Read a vector of field metadata
    std::vector<IndexDataFieldMeta> readFieldMeta() override
    {
        std::vector<IndexDataFieldMeta> results;
        auto file = istream_constructor(
            path_prefix + INDEX_META_FILENAME, std::ios::in);

        // save IStream content to stringstream for json parsing
        std::stringstream str_stream;
        std::vector<char> buffer(META_BUFFER_SIZE);
        while (!file->eof())
        {
            file->read(&buffer[0], META_BUFFER_SIZE);
            str_stream.read(&buffer[0], file->gcount());
        }
        // start reading from the beginning
        str_stream.seekg(0);

        std::string line;
        while (std::getline(str_stream, line))
        {
            json j = json::parse(line);
            if (j)
            {
                results.push_back(IndexFieldMetaIO::parse(j));
            }
        }
        return results;
    }

    bool hasFieldData(const std::string & name) override
    {
        std::string file_name
            = path_prefix + INDEX_DATA_FILED_FILENAME_FMT(name);
        return std::filesystem::exists(file_name);
    }

    /// Get the input stream for field data
    std::shared_ptr<IS>
    getFieldDataInputStream(const std::string & name) override
    {
        std::string file_name
            = path_prefix + INDEX_DATA_FILED_FILENAME_FMT(name);
        SI_THROW_IF_NOT_FMT(
            std::filesystem::exists(file_name),
            ErrorCode::CANNOT_OPEN_FILE,
            "getFieldDataInputStream name=%s, file_name=%s doesn't exist",
            name.c_str(),
            file_name.c_str());
        auto istream = istream_constructor(file_name, std::ios::binary);
        SI_THROW_IF_NOT_FMT(
            istream->is_open() && !istream->fail(),
            ErrorCode::CANNOT_OPEN_FILE,
            "getFieldDataInputStream error, file_name=%s failed to open",
            file_name.c_str());

        SI_LOG_INFO(
            "IndexDataFileReader::getFieldDataInputStream name={}, "
            "file_name={}",
            name,
            file_name);
        return istream;
    }

    bool readFieldChecksum(const std::string & name, size_t & checksum) override
    {
        SI_LOG_INFO("IndexDataFileReader::readFieldChecksum name={}", name);
        auto istream = istream_constructor(
            path_prefix + INDEX_CHECKSUM_FILED_FILENAME_FMT(name),
            std::ios::binary);
        if (!istream->is_open())
        {
            return false;
        }
        istream->read(reinterpret_cast<char *>(&checksum), sizeof(size_t));
        // whether the read is successful
        return istream->gcount() == sizeof(size_t);
    }
    ~IndexDataFileReader() override = default;

private:
    std::string path_prefix;
    IStreamConstructor istream_constructor;
};

/// @brief Write index files with local files.
template <typename OS>
class IndexDataFileWriter final : public IndexDataWriter<OS>
{
public:
    using OStreamConstructor = std::function<std::shared_ptr<OS>(
        const std::string & name, std::ios::openmode)>;

    explicit IndexDataFileWriter(
        const std::string & path_prefix_,
        OStreamConstructor ostream_constructor_) :
        path_prefix(path_prefix_),
        ostream_constructor(std::move(ostream_constructor_))
    {
    }

    /// Write a vector of field metadata
    bool writeFieldMeta(const std::vector<IndexDataFieldMeta> & fields) override
    {
        auto file = ostream_constructor(
            path_prefix + INDEX_META_FILENAME, std::ios::out);
        for (const auto & f : fields)
        {
            auto meta_str = IndexFieldMetaIO::dump(f);
            file->write(meta_str.c_str(), meta_str.size());
            file->write("\n", 1);
        }
        file->close();
        return true;
    }

    /// Get the input stream for field data
    std::shared_ptr<OS>
    getFieldDataOutputStream(const std::string & name) override
    {
        SI_LOG_INFO(
            "IndexDataFileWriter::getFieldDataOutputStream name={}", name);
        return ostream_constructor(
            path_prefix + INDEX_DATA_FILED_FILENAME_FMT(name),
            std::ios::binary);
    }

    void writeFieldChecksum(const std::string & name, size_t checksum) override
    {
        SI_LOG_INFO(
            "IndexDataFileWriter::writeFieldChecksum name={}, checksum={}",
            name,
            checksum);
        auto ostream = ostream_constructor(
            path_prefix + INDEX_CHECKSUM_FILED_FILENAME_FMT(name),
            std::ios::binary);
        ostream->write(reinterpret_cast<char *>(&checksum), sizeof(size_t));
        ostream->close();
    }

    ~IndexDataFileWriter() override = default;

private:
    std::string path_prefix;
    OStreamConstructor ostream_constructor;
};

/// @brief Input stream backed by local files and `ifstream` internally.
class FileBasedIStream : public AbstractIStream
{
public:
    explicit FileBasedIStream(
        const std::string & file_name_,
        std::ios::openmode mode = std::ios::in) :
        file_name(file_name_), istream(file_name_, mode)
    {
        SI_LOG_INFO("Creating FileBasedIStream, file_name {}", file_name);
    }

    inline AbstractIStream & read(char * s, std::streamsize count) override
    {
        istream.read(s, count);
        return *this;
    }

    bool fail() const override { return istream.fail(); }

    bool eof() const override { return istream.eof(); }

    bool is_open() const override { return istream.is_open(); }

    std::streamsize gcount() const override { return istream.gcount(); }

    explicit operator bool() const override
    {
        return static_cast<bool>(istream);
    }

    /// Be more compatible with ifstream
    AbstractIStream &
    seekg(std::streampos offset, std::ios::seekdir dir) override
    {
        // beg: 0, cur: 1, end: 2
        SI_LOG_WARNING(
            "FileBasedIStream::seekg offset={}, dir={} "
            "might not be supported by DatabaseIStream",
            offset,
            static_cast<int>(dir));
        istream.seekg(offset, dir);
        return *this;
    }

private:
    std::string file_name;
    std::ifstream istream;
};

/// @brief Output stream backed by local files and `ofstream` internally.
class FileBasedOStream : public AbstractOStream
{
public:
    explicit FileBasedOStream(
        const std::string & file_name_,
        std::ios::openmode mode = std::ios::out) :
        file_name(file_name_), ostream(file_name_, mode)
    {
        SI_LOG_INFO("Creating FileBasedOStream, file_name {}", file_name);
    }

    AbstractOStream & write(const char * s, std::streamsize count) override
    {
        ostream.write(s, count);
        return *this;
    }

    bool good() override { return ostream.good(); }

    void close() override { ostream.close(); }

    /// Be more compatible with ofstream
    AbstractOStream &
    seekp(std::streampos offset, std::ios::seekdir dir) override
    {
        SI_LOG_WARNING(
            "FileBasedOStream::seekp offset={} dir={}, "
            "might not be supported by DatabaseIStram",
            offset,
            static_cast<int>(dir));
        ostream.seekp(offset, dir);
        return *this;
    }

private:
    std::string file_name;
    std::ofstream ostream;
};

}
