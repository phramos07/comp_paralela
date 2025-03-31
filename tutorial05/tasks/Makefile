# Use standard clang with preprocessor flags
CC=/opt/homebrew/opt/llvm@18/bin/clang -w 

# System include paths
SYSINCLUDE=-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
SDK_PATH=$(shell xcrun --show-sdk-path)

# OpenMP flags and includes
OMPFLAGS=-fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp

SRC=$(wildcard *.c)
TARGET=$(SRC:.c=.out)

# Basic compilation with -O0
noopt: $(SRC)
	$(CC) -O0 $(SYSINCLUDE) -isysroot $(SDK_PATH) $< -o $(TARGET) && ./$(TARGET)

# Basic compilation with -O3  
opt: $(SRC)
	$(CC) -O3 $(SYSINCLUDE) -isysroot $(SDK_PATH) $< -o $(TARGET) && ./$(TARGET)

# OpenMP compilation with -O0
parallelnoopt: $(SRC)
	$(CC) -O0 $(SYSINCLUDE) -isysroot $(SDK_PATH) $(OMPFLAGS) $< -o $(TARGET) && ./$(TARGET)

# OpenMP compilation with -O3
parallelopt: $(SRC)
	$(CC) -O3 $(SYSINCLUDE) -isysroot $(SDK_PATH) $(OMPFLAGS) $< -o $(TARGET) && ./$(TARGET)

clean:
	rm -f *.out