# Creating a Custom Test Runner Agent in Claude Code

This guide shows you how to create a reusable test runner agent using Claude Code's `/agents` command.

## Prerequisites
- Claude Code installed and running in interactive mode
- Access to the `/agents` command in your Claude Code session

## Step-by-Step Instructions

### 1. Check Available Agent Commands
First, explore what agent commands are available:
```
/agents help
```
This will show you all available subcommands for managing agents.

### 2. Create the Test Runner Agent
Create a new agent called "test-runner":
```
/agents create test-runner
```

### 3. Configure the Agent
When prompted for the agent configuration, provide the following:

**Name:** `test-runner`

**Description:** `Compiles and runs unit test projects for the Milo codebase`

**Prompt/Instructions:**
```
You are a specialized test runner agent for the Milo project. When given a test project name:

1. COMPILE THE TEST:
   - Use MSBuild located at: "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
   - Project path pattern: unittest/builds/VisualStudio2022/projects/{TEST_NAME}.vcxproj
   - Build configuration: /p:Configuration=Debug /p:Platform=x64
   - Full command: cmd.exe /c "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" unittest/builds/VisualStudio2022/projects/{TEST_NAME}.vcxproj /p:Configuration=Debug /p:Platform=x64
   - Report any compilation errors in detail

2. RUN THE TEST (if compilation succeeds):
   - Executable location: builds/bin/Debug-windows-x86_64/{TEST_NAME}/{TEST_NAME}.exe
   - Full command: cmd.exe /c builds\bin\Debug-windows-x86_64\{TEST_NAME}\{TEST_NAME}.exe
   - Capture ALL output including test results and any debug messages

3. PROVIDE A DETAILED SUMMARY:
   - Compilation status: ✅ SUCCESS or ❌ FAILED
   - If compilation failed: Show error messages and line numbers
   - Test execution results:
     * Total test cases run
     * Number passed
     * Number failed
     * Number skipped
   - For failed tests: Include test name, file location, line number, and error message
   - Overall status: SUCCESS or FAILURE
   - Execution time for both compilation and testing

Replace {TEST_NAME} with the actual test name provided (e.g., ShockerModelTest, GeometryTest, HelloTest).

Example: If input is "ShockerModelTest", compile and run ShockerModelTest.vcxproj and ShockerModelTest.exe
```

### 4. Save the Agent
The agent should save automatically after configuration. Confirm the save when prompted.

### 5. Using the Test Runner Agent

#### Run a specific test:
```
/agents run test-runner ShockerModelTest
```

#### Run other tests:
```
/agents run test-runner GeometryTest
/agents run test-runner HelloTest
```

### 6. Managing Your Agents

#### List all custom agents:
```
/agents list
```

#### Edit the test runner agent:
```
/agents edit test-runner
```

#### Delete an agent (if needed):
```
/agents delete test-runner
```

#### View agent details:
```
/agents show test-runner
```

## Advanced Usage

### Creating a Test-All Agent
You could create another agent that runs all tests:

```
/agents create test-all
```

**Prompt:**
```
Run all unit tests in unittest/builds/VisualStudio2022/projects:
1. List all .vcxproj files in the directory
2. For each test project:
   - Compile it
   - Run it if compilation succeeds
   - Track results
3. Provide a summary table:
   | Test Name | Compiled | Tests Run | Passed | Failed |
```

### Creating Test Agents for Specific Configurations

#### Release Mode Tester:
```
/agents create test-runner-release
```
Configure it to use `/p:Configuration=Release` instead of Debug.

#### Specific Test Suite Runner:
```
/agents create test-shocker-suite
```
Configure it to run all Shocker-related tests (ShockerModelTest, ShockerHandlerTest, etc.)

## Troubleshooting

### If the agent doesn't work:
1. Check that the test project exists: `unittest/builds/VisualStudio2022/projects/{TEST_NAME}.vcxproj`
2. Verify MSBuild path is correct for your system
3. Ensure the test executable was built to the expected location
4. Use `/agents edit test-runner` to modify the agent configuration

### Common Issues:
- **PDB locked error**: The test might still be running. Wait and retry.
- **Project not found**: Check the exact project name and path.
- **MSBuild not found**: Verify Visual Studio 2022 installation path.

## Benefits of Using Custom Agents

1. **Consistency**: Same test process every time
2. **Speed**: No need to remember commands or paths
3. **Automation**: Agent handles compilation and execution
4. **Reporting**: Consistent, detailed test results
5. **Reusability**: Use across different sessions
6. **Shareable**: Export and share agent configurations with team

## Next Steps

After creating your test runner agent:
1. Test it with different unit test projects
2. Create specialized agents for different test scenarios
3. Consider creating agents for other repetitive tasks like:
   - Code formatting checks
   - Build verification
   - Documentation generation
   - Performance profiling

## Example Session

```bash
> /agents run test-runner ShockerModelTest

🤖 Test Runner Agent Starting...
📦 Compiling ShockerModelTest...
✅ Compilation SUCCESS (15.4 seconds)
🧪 Running tests...
✅ All tests PASSED!
   - Test cases: 6/6 passed
   - Assertions: 62/62 passed
   - Time: 0.23 seconds
🎉 Overall: SUCCESS
```