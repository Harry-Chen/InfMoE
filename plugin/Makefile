.PHONY: all compile clean distclean

ifeq ($(shell which meson),)
    $(error Please install meson(>=0.50.0) first!)
endif

ifeq ($(shell which ninja),)
    $(error Please install ninja first!)
endif

TENSORRT_PREFIX ?= /usr
CUDNN_PREFIX ?= /usr

all: compile

builddir:
	meson setup builddir -DWITH_TENSORRT=${TENSORRT_PREFIX} -DWITH_CUDNN=${CUDNN_PREFIX}

compile: builddir
	ninja -C builddir

clean:
	ninja -C builddir clean

distclean:
	rm -rf builddir
