# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/pcd/vscodes/pcd_nricp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pcd/vscodes/pcd_nricp/build

# Include any dependencies generated for this target.
include CMakeFiles/dfm_transf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dfm_transf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dfm_transf.dir/flags.make

CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.o: CMakeFiles/dfm_transf.dir/flags.make
CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.o: ../deformation_transfer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pcd/vscodes/pcd_nricp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.o -c /home/pcd/vscodes/pcd_nricp/deformation_transfer.cpp

CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pcd/vscodes/pcd_nricp/deformation_transfer.cpp > CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.i

CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pcd/vscodes/pcd_nricp/deformation_transfer.cpp -o CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.s

# Object files for target dfm_transf
dfm_transf_OBJECTS = \
"CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.o"

# External object files for target dfm_transf
dfm_transf_EXTERNAL_OBJECTS =

dfm_transf: CMakeFiles/dfm_transf.dir/deformation_transfer.cpp.o
dfm_transf: CMakeFiles/dfm_transf.dir/build.make
dfm_transf: ../3rdparty/trimesh2/lib.Linux64/libtrimesh.a
dfm_transf: /usr/lib/x86_64-linux-gnu/libamd.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libcamd.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libccolamd.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libcolamd.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libcholmod.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libspqr.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libldl.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libbtf.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libklu.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libcxsparse.so
dfm_transf: /usr/lib/x86_64-linux-gnu/libumfpack.so
dfm_transf: /usr/lib/x86_64-linux-gnu/liblz4.so
dfm_transf: CMakeFiles/dfm_transf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pcd/vscodes/pcd_nricp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dfm_transf"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dfm_transf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dfm_transf.dir/build: dfm_transf

.PHONY : CMakeFiles/dfm_transf.dir/build

CMakeFiles/dfm_transf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dfm_transf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dfm_transf.dir/clean

CMakeFiles/dfm_transf.dir/depend:
	cd /home/pcd/vscodes/pcd_nricp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pcd/vscodes/pcd_nricp /home/pcd/vscodes/pcd_nricp /home/pcd/vscodes/pcd_nricp/build /home/pcd/vscodes/pcd_nricp/build /home/pcd/vscodes/pcd_nricp/build/CMakeFiles/dfm_transf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dfm_transf.dir/depend

