# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu

# Include any dependencies generated for this target.
include CMakeFiles/icp_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/icp_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/icp_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/icp_cpu.dir/flags.make

CMakeFiles/icp_cpu.dir/icp.cpp.o: CMakeFiles/icp_cpu.dir/flags.make
CMakeFiles/icp_cpu.dir/icp.cpp.o: icp.cpp
CMakeFiles/icp_cpu.dir/icp.cpp.o: CMakeFiles/icp_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/icp_cpu.dir/icp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/icp_cpu.dir/icp.cpp.o -MF CMakeFiles/icp_cpu.dir/icp.cpp.o.d -o CMakeFiles/icp_cpu.dir/icp.cpp.o -c /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/icp.cpp

CMakeFiles/icp_cpu.dir/icp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/icp_cpu.dir/icp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/icp.cpp > CMakeFiles/icp_cpu.dir/icp.cpp.i

CMakeFiles/icp_cpu.dir/icp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/icp_cpu.dir/icp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/icp.cpp -o CMakeFiles/icp_cpu.dir/icp.cpp.s

CMakeFiles/icp_cpu.dir/edge.cpp.o: CMakeFiles/icp_cpu.dir/flags.make
CMakeFiles/icp_cpu.dir/edge.cpp.o: edge.cpp
CMakeFiles/icp_cpu.dir/edge.cpp.o: CMakeFiles/icp_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/icp_cpu.dir/edge.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/icp_cpu.dir/edge.cpp.o -MF CMakeFiles/icp_cpu.dir/edge.cpp.o.d -o CMakeFiles/icp_cpu.dir/edge.cpp.o -c /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/edge.cpp

CMakeFiles/icp_cpu.dir/edge.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/icp_cpu.dir/edge.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/edge.cpp > CMakeFiles/icp_cpu.dir/edge.cpp.i

CMakeFiles/icp_cpu.dir/edge.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/icp_cpu.dir/edge.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/edge.cpp -o CMakeFiles/icp_cpu.dir/edge.cpp.s

# Object files for target icp_cpu
icp_cpu_OBJECTS = \
"CMakeFiles/icp_cpu.dir/icp.cpp.o" \
"CMakeFiles/icp_cpu.dir/edge.cpp.o"

# External object files for target icp_cpu
icp_cpu_EXTERNAL_OBJECTS =

icp_cpu: CMakeFiles/icp_cpu.dir/icp.cpp.o
icp_cpu: CMakeFiles/icp_cpu.dir/edge.cpp.o
icp_cpu: CMakeFiles/icp_cpu.dir/build.make
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_people.so
icp_cpu: /usr/lib/libOpenNI.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_features.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_search.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_io.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libpng.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libz.so
icp_cpu: /usr/lib/libOpenNI.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libfreetype.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libGLEW.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libX11.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
icp_cpu: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
icp_cpu: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
icp_cpu: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
icp_cpu: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libpcl_common.so
icp_cpu: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
icp_cpu: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
icp_cpu: CMakeFiles/icp_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable icp_cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/icp_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/icp_cpu.dir/build: icp_cpu
.PHONY : CMakeFiles/icp_cpu.dir/build

CMakeFiles/icp_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/icp_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/icp_cpu.dir/clean

CMakeFiles/icp_cpu.dir/depend:
	cd /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu /mnt/c/Users/Code/HardAcl/ICP-HardwareAccel/icp_cpu/CMakeFiles/icp_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/icp_cpu.dir/depend

