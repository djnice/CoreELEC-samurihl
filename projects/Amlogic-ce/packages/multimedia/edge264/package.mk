# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026-present Team CoreELEC (https://coreelec.org)

PKG_NAME="edge264"
PKG_VERSION="5757e71f3decbeced1150e742e802d263fbd0df4"
PKG_SHA256="bcf3b26581312c64f58ed37866d8cded2a3a934c997d72ab7eb6abc0790db496"
PKG_LICENSE="BSD"
PKG_SITE="https://github.com/jens-duttke/edge264-mvc"
PKG_URL="https://github.com/jens-duttke/edge264-mvc/archive/${PKG_VERSION}.tar.gz"
PKG_SOURCE_DIR="edge264-mvc-${PKG_VERSION}"
PKG_DEPENDS_TARGET="toolchain"
PKG_LONGDESC="edge264-mvc: H.264 and 3D MVC software decoder (software MVC decoding in Kodi)"
PKG_TOOLCHAIN="manual"

make_target() {
  # static, no log variant; the Makefile's native -march is for host builds,
  # and -O3 has to come after the -O2 of the target flags
  make -C ${PKG_BUILD} OS=linux STATIC=yes VARIANTS= BUILDTEST=no _BASE_ARCH=        CC="${CC}" AR="${AR}" CFLAGS="${CFLAGS} -O3"
}

makeinstall_target() {
  mkdir -p ${SYSROOT_PREFIX}/usr/{lib,include}
  cp ${PKG_BUILD}/libedge264.a ${SYSROOT_PREFIX}/usr/lib
  cp ${PKG_BUILD}/edge264.h ${SYSROOT_PREFIX}/usr/include
}
