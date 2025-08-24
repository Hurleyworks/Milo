# test

Compile and run unit tests

## Usage

```
/test                    # Run all unit tests
/test CgModel2Shocker    # Run specific test
```

## Description

This command builds and runs unit tests for the Milo project. It will:
- Compile the unit test solution
- Run the test executables
- Report build errors if compilation fails
- Report test failures with detailed output

## Examples

```
/test
```
Builds and runs all unit tests in the unittest directory.

```
/test ShockerModelTest
```
Builds and runs only the ShockerModelTest.

## Implementation

When this command is invoked, Claude should:
1. Change to the unittest directory
2. Run the test.bat script with any provided arguments
3. If build fails, read unittest/build_errors.txt and fix the errors
4. If tests fail, read unittest/test_results.txt and analyze the failures