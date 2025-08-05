#include "Jahley.h"

const std::string APP_NAME = "HelloTest";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <sabi_core/sabi_core.h>

// Simple test cases
TEST_CASE("Basic arithmetic operations") {
    SUBCASE("Addition works correctly") {
        CHECK(2 + 3 == 5);
        CHECK(-1 + 1 == 0);
        CHECK(0 + 0 == 0);
        CHECK(100 + 200 == 300);
    }
    
    SUBCASE("Multiplication works correctly") {
        CHECK(2 * 3 == 6);
        CHECK(-2 * 3 == -6);
        CHECK(0 * 100 == 0);
        CHECK(7 * 8 == 56);
    }
}

TEST_CASE("String operations") {
    std::string hello = "Hello";
    std::string world = "World";
    
    CHECK(hello.length() == 5);
    CHECK(world.length() == 5);
    
    std::string combined = hello + " " + world;
    CHECK(combined == "Hello World");
    CHECK(combined.length() == 11);
}

TEST_CASE("Vector operations") {
    std::vector<int> numbers;
    
    CHECK(numbers.empty() == true);
    CHECK(numbers.size() == 0);
    
    numbers.push_back(10);
    numbers.push_back(20);
    numbers.push_back(30);
    
    CHECK(numbers.size() == 3);
    CHECK(numbers[0] == 10);
    CHECK(numbers[1] == 20);
    CHECK(numbers[2] == 30);
}

TEST_CASE("Sabi core functionality") {
    // Test that we can use types from sabi_core
    Eigen::Vector3f vec(1.0f, 2.0f, 3.0f);
    CHECK(vec.x() == doctest::Approx(1.0f));
    CHECK(vec.y() == doctest::Approx(2.0f));
    CHECK(vec.z() == doctest::Approx(3.0f));
    
    // Test vector operations
    Eigen::Vector3f vec2(4.0f, 5.0f, 6.0f);
    Eigen::Vector3f sum = vec + vec2;
    CHECK(sum.x() == doctest::Approx(5.0f));
    CHECK(sum.y() == doctest::Approx(7.0f));
    CHECK(sum.z() == doctest::Approx(9.0f));
}

class Application : public Jahley::App
{
 public:
    Application (DesktopWindowSettings settings = DesktopWindowSettings(), bool windowApp = false) :
        Jahley::App()
    {
        doctest::Context().run();
    }

 private:
};

Jahley::App* Jahley::CreateApplication()
{
    return new Application();
}
