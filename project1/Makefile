CMAKE_BUILD_TYPE ?= Debug

.PHONY: main
main:
	mkdir -p build
	cd build && cmake ..
	make -C build/ -j
	mv build/main .

.PHONY: clean
clean:
	rm -r build main
