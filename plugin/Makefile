.PHONY: all compile clean distclean

ifeq ($(shell which meson),)
    $(error Please install meson(>=0.50.0) first!)
endif

ifeq ($(shell which ninja),)
    $(error Please install ninja first!)
endif

DEBUG ?= true
TENSORRT_PREFIX ?= /usr
CUDNN_PREFIX ?= /usr
BUILDDIR ?= builddir

ifeq (${DEBUG}, true)
	BUILDTYPE = debugoptimized
else
	BUILDTYPE = release
endif

all: compile

${BUILDDIR}:
	meson setup "${BUILDDIR}" -DWITH_TENSORRT=${TENSORRT_PREFIX} -DWITH_CUDNN=${CUDNN_PREFIX} --buildtype ${BUILDTYPE}

compile: ${BUILDDIR}
	ninja -C "${BUILDDIR}"

clean:
	ninja -C "${BUILDDIR}" clean

distclean:
	rm -rf "${BUILDDIR}"
