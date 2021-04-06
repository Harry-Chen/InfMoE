.PHONY: all compile clean

ifeq ($(shell which meson),)
    $(error Please install meson(>=0.50.0) first!)
endif

ifeq ($(shell which ninja),)
    $(error Please install ninja first!)
endif

TENSORRT_PREFIX ?= /usr

all: builddir compile

builddir:
	meson setup build -DWITH_TENSORRT=${TENSORRT_PREFIX}

compile: builddir
	ninja -C build

clean: builddir
	ninja -C build clean
