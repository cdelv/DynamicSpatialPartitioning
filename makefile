# Compiler and flags
CXX         := g++
CXXFLAGS    := -Wall -std=c++11 -O3 -march=native -I src -fopenmp

# Directories
SRCDIR      := src
OBJDIR      := objects

# Main source file (in the main directory)
MAIN        := main.cpp

# Source files in src/ (we exclude SmallList.cpp because its implementation
# is handled via explicit instantiation in instantiations.cpp)
SRC_FILES   := $(wildcard $(SRCDIR)/*.cpp)

# Object files: main.cpp compiles to objects/main.o,
# and each .cpp in src compiles to objects/<name>.o
MAIN_OBJ    := $(OBJDIR)/main.o
SRC_OBJS    := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC_FILES))
OBJECTS     := $(MAIN_OBJ) $(SRC_OBJS)

# Default target: produce a.out in the main directory.
a.out: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o a.out $(OBJECTS)

# Rule to compile main.cpp
$(OBJDIR)/main.o: main.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile source files in src/
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(OBJDIR) a.out