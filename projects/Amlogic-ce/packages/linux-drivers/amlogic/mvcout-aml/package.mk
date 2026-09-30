# SPDX-License-Identifier: GPL-2.0-or-later

PKG_NAME="mvcout-aml"
PKG_VERSION="0.1"
PKG_LICENSE="GPL"
PKG_SITE="https://coreelec.org"
PKG_URL=""
PKG_DEPENDS_TARGET="toolchain linux"
PKG_NEED_UNPACK="${LINUX_DEPENDS}"
PKG_LONGDESC="vframe provider for software decoded 3D MVC pictures (Amlogic S5)"
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

  # loaded at boot; Kodi decodes 3D MVC in software when /dev/mvcout exists
  mkdir -p ${INSTALL}/usr/lib/modules-load.d
    echo "mvcout" > ${INSTALL}/usr/lib/modules-load.d/mvcout.conf
}
