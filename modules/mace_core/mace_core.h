#pragma once

#include <unordered_map>
#include <unordered_set>
#include <array>
#include <queue>
#include <stack>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <random>
#include <chrono>
#include <thread>
#include <ctime>
#include <string>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <limits>
#include <algorithm>
#include <functional>
#include <stdint.h>
#include <any>
#include <filesystem>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <variant>
#include <future>
#include <semaphore>
#include <concepts>
#include <numbers>
#include <variant>

#ifdef __clang__
#include <experimental/coroutine>
#define COROUTINE_NAMESPACE std::experimental
#else
#include <coroutine>
#define COROUTINE_NAMESPACE std
#endif

// eigen math
#include <linalg/eigen34/Eigen/Dense>

// libassert and cpptrace
#include <cpptrace/cpptrace.hpp>
#include <libassert/assert.hpp>

// nano signal and slots
#include <nano_signal/nano_signal_slot.hpp>
#include <nano_signal/nano_mutex.hpp>
using Observer = Nano::Observer<>;

// openimageio
#ifdef USE_OIIO
#include <OpenImageIO/thread.h>
#include <OpenImageIO/unordered_map_concurrent.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagecache.h>
#endif

// g3log
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>

// moody camel lock free queue
#include <concurrent/concurrentqueue.h>

// random
#include <random/include/random.hpp>
using RandoM = effolkronium::random_static;

// BS_thread_pool version 4. I had to modify it
#include <BSthread/BS_thread_pool.hpp>
#include <BSthread/BS_thread_pool_utils.hpp>

// BinaryReader/Writer
#include <BinaryWriter.h>
#include <BinaryReader.h>

// json
#include <json/json.hpp>
using nlohmann::json;

// tinyformat
#include <tinyformat/tinyformat.h>

// less typing
namespace fs = std::filesystem;

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET SocketHandle;
#define INVALID_SOCKET_HANDLE INVALID_SOCKET
#define SOCKET_ERROR_HANDLE SOCKET_ERROR
#define CloseSocketHandle closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
typedef int SocketHandle;
#define INVALID_SOCKET_HANDLE (-1)
#define SOCKET_ERROR_HANDLE (-1)
#define CloseSocketHandle close
#endif

// some useful tools and defines outside mace namespace
#include "excludeFromBuild/basics/Defaults.h"
#include "excludeFromBuild/basics/Util.h"

namespace mace
{
// basics
#include "excludeFromBuild/basics/StringUtil.h"
#include "excludeFromBuild/basics/InputEvent.h"

} // namespace mace
