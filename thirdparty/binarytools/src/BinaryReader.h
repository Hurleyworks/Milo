#pragma once
#include "MemoryBuffer.h"
#include <fstream>
#include <string>
#include "Span.h" // Include our custom Span instead of <span>
#include <vector>

struct MemoryBuffer;

class BinaryReader
{
 public:
    BinaryReader (std::string_view inputPath);
    BinaryReader (char* buffer, uint32_t sizeInBytes);
    BinaryReader (Span<uint8_t> buffer); // Changed from std::span to our custom Span
    ~BinaryReader();


    [[nodiscard]] uint8_t ReadUint8();
    [[nodiscard]] uint16_t ReadUint16();
    [[nodiscard]] uint32_t ReadUint32();
    [[nodiscard]] uint64_t ReadUint64();

    [[nodiscard]] int8_t ReadInt8();
    [[nodiscard]] int16_t ReadInt16();
    [[nodiscard]] int32_t ReadInt32();
    [[nodiscard]] int64_t ReadInt64();

    [[nodiscard]] char ReadChar();
    [[nodiscard]] wchar_t ReadCharWide();
    [[nodiscard]] std::string ReadNullTerminatedString();
    [[nodiscard]] std::string ReadFixedLengthString(size_t length);
    [[nodiscard]] std::wstring ReadNullTerminatedStringWide();
    [[nodiscard]] std::wstring ReadFixedLengthStringWide(size_t length);
    [[nodiscard]] std::vector<std::string> ReadSizedStringList(size_t listSize);
    [[nodiscard]] char PeekChar();
    [[nodiscard]] uint32_t PeekUint32();
    [[nodiscard]] wchar_t PeekCharWide();

    [[nodiscard]] float ReadFloat();
    [[nodiscard]] double ReadDouble();

    void ReadToMemory(void* destination, size_t size);

    void SeekBeg(size_t absoluteOffset);
    void SeekCur(size_t relativeOffset);
    void SeekReverse(size_t relativeOffset); //Move backwards from the current stream position
    void Skip(size_t bytesToSkip);
    size_t Align(size_t alignmentValue = 2048);

    size_t Position() const;
    size_t Length();

private:
    std::istream* stream_ = nullptr;
    basic_memstreambuf* buffer_ = nullptr;
};

