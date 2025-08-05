#include "Jahley.h"

const std::string APP_NAME = "HelloWorld";


class Application : public Jahley::App
{
public:
	Application() :
		Jahley::App()
	{
		try
		{
			LOG(INFO) << "Hello World";
		}
		catch (std::exception& e)
		{
			LOG(WARNING) << e.what();
		}
	}

	~Application()
	{
	}

	void onCrash() override
	{
	}

private:
};

Jahley::App* Jahley::CreateApplication()
{
	return new Application();
}
