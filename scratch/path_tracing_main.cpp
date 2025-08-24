#include "Application.h"

int main(int argc, const char* argv[]) {
	try {
		PathTracingApp app;
		return app.run(argc, argv);
	}
	catch (const std::exception& ex) {
		hpprintf("Error: %s\n", ex.what());
		return -1;
	}
}
