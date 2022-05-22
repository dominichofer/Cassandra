#pragma once
#include <filesystem>

template <typename Base>
class Filebased : public Base
{
public:
	std::filesystem::path file;
private:
	bool auto_write_back;

	static Base DeserializeBase(std::filesystem::path file)
	{
		std::fstream stream(file, std::ios::binary | std::ios::in);
		return Deserialize<Base>(stream);
	}
public:
	Filebased() = default;
	Filebased(std::filesystem::path file, bool auto_write_back = false)
		: Base(DeserializeBase(file))
		, file(file)
		, auto_write_back(auto_write_back)
	{}
	Filebased(std::filesystem::path file, Base base, bool auto_write_back = false)
		: Base(std::move(base))
		, file(file)
		, auto_write_back(auto_write_back)
	{}
	Filebased& operator=(const Base& o) { *this = Filebased(file, o, auto_write_back); return *this; }
	~Filebased() { if (auto_write_back) WriteBack(); }
	static Filebased<Base> WithAutoWriteBack(std::filesystem::path file) { return { file, true }; }

	void SaveTo(std::filesystem::path file) const
	{
		std::fstream stream(file, std::ios::binary | std::ios::out);
		Serialize(static_cast<const Base&>(*this), stream);
	}
	void WriteBack() const { SaveTo(file); }

	bool AutoWriteBack() const { return auto_write_back; }
	void AutoWriteBack(bool value) { auto_write_back = value; }
	void EnableAutoWriteBack() { auto_write_back = true; }
	void DisableAutoWriteBack() { auto_write_back = false; }
};
