
#pragma once

#include "../AppConfig.h"

// This C++ class LogHandler takes two callback functions as arguments in its constructor :

// FatalErrorCallback is a type alias for a std::function that takes a g3::FatalMessagePtr object as a parameter and returns void.
// This callback is meant to handle fatal error messages that may be generated during program execution.

// PreCrashCallback is a type alias for a std::function that takes no parameters and returns void.
// This callback is meant to be called before the program crashes, possibly allowing for some 
// cleanup or other actions to be taken.
// 
// The class has a single private member variable, std::unique_ptr<g3::LogWorker> worker, 
// which is a pointer to a LogWorker object from the g3 logging library.

// The purpose of this class appears to be to handle logging and errors in a program
// by allowing the user to define custom callbacks that are called in the event of a 
// fatal error or prior to a crash.The LogHandler class is responsible for managing
// the LogWorker object that handles the actual logging and error reporting.

using FatalErrorCallback = std::function<void (g3::FatalMessagePtr)>;
using PreCrashCallback = std::function<void()>;

class LogHandler
{
 public:
    LogHandler (FatalErrorCallback crashCallback, PreCrashCallback precrashCallback);
    ~LogHandler() = default;

 private:
    std::unique_ptr<g3::LogWorker> worker;

}; // end class LogHandler