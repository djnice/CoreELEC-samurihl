# SPDX-License-Identifier: GPL-2.0-or-later

PKG_NAME="dvfel-aml"
PKG_VERSION="0.1"
PKG_LICENSE="GPL"
PKG_SITE="https://coreelec.org"
PKG_URL=""
PKG_DEPENDS_TARGET="toolchain linux"
PKG_NEED_UNPACK="${LINUX_DEPENDS}"
PKG_LONGDESC="Dolby Vision FEL composition vfm node for Amlogic S5 (S928X)"
PKG_TOOLCHAIN="manual"

pre_make_target() {
  unset LDFLAGS
}

make_target() {
  kernel_make -C $(kernel_path) M=${PKG_BUILD}
}

makeinstall_target() {
  mkdir -p ${INSTALL}/$(get_full_module_dir)/${PKG_NAME}
    cp ${PKG_BUILD}/*.ko ${INSTALL}/$(get_full_module_dir)/${PKG_NAME}
}
