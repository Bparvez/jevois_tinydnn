# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jevois/objdetect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jevois/objdetect/hbuild

# Include any dependencies generated for this target.
include CMakeFiles/ObjDetect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ObjDetect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ObjDetect.dir/flags.make

CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o: CMakeFiles/ObjDetect.dir/flags.make
CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o: ../src/Modules/ObjDetect/ObjDetect.C
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jevois/objdetect/hbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o -c /home/jevois/objdetect/src/Modules/ObjDetect/ObjDetect.C

CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jevois/objdetect/src/Modules/ObjDetect/ObjDetect.C > CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.i

CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jevois/objdetect/src/Modules/ObjDetect/ObjDetect.C -o CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.s

CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.requires:

.PHONY : CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.requires

CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.provides: CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.requires
	$(MAKE) -f CMakeFiles/ObjDetect.dir/build.make CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.provides.build
.PHONY : CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.provides

CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.provides.build: CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o


# Object files for target ObjDetect
ObjDetect_OBJECTS = \
"CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o"

# External object files for target ObjDetect
ObjDetect_EXTERNAL_OBJECTS =

ObjDetect.so: CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o
ObjDetect.so: CMakeFiles/ObjDetect.dir/build.make
ObjDetect.so: CMakeFiles/ObjDetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jevois/objdetect/hbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ObjDetect.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ObjDetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ObjDetect.dir/build: ObjDetect.so

.PHONY : CMakeFiles/ObjDetect.dir/build

CMakeFiles/ObjDetect.dir/requires: CMakeFiles/ObjDetect.dir/src/Modules/ObjDetect/ObjDetect.C.o.requires

.PHONY : CMakeFiles/ObjDetect.dir/requires

CMakeFiles/ObjDetect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ObjDetect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ObjDetect.dir/clean

CMakeFiles/ObjDetect.dir/depend:
	cd /home/jevois/objdetect/hbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jevois/objdetect /home/jevois/objdetect /home/jevois/objdetect/hbuild /home/jevois/objdetect/hbuild /home/jevois/objdetect/hbuild/CMakeFiles/ObjDetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ObjDetect.dir/depend

