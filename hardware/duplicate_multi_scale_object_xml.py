import os
import glob
from lxml import etree

# 要处理的 scale 比例
scales = [0.06, 0.08, 0.10, 0.12]

# 获取所有 coacd.xml 文件路径
object_asset = "/home/hand/intern/BODex/src/curobo/content/assets/object/DGN_2k"
coacd_files = glob.glob(os.path.join(object_asset, "processed_data/**/urdf/coacd.xml"), recursive=True)

for obj_dir in coacd_files:
    source_xml = obj_dir
    if not os.path.exists(source_xml):
        continue

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(source_xml, parser)
    root = tree.getroot()

    # 找到原始的 <asset> 和所有 <mesh>
    asset = root.find("asset")
    if asset is None:
        asset = etree.SubElement(root, "asset")

    mesh_elements = asset.findall("mesh")

    # 收集所有 geom（无论嵌套在哪里）
    geoms = root.findall(".//geom")

    for scale in scales:
        scale_str = f"{int(scale*100):03d}"
        new_root = etree.Element("mujoco", model=f"scaled_{scale_str}")

        # asset with scaled mesh
        new_asset = etree.SubElement(new_root, "asset")
        for mesh in mesh_elements:
            new_mesh = etree.SubElement(new_asset, "mesh", attrib=mesh.attrib)
            new_mesh.attrib["scale"] = f"{scale} {scale} {scale}"

        # worldbody with one body
        worldbody = etree.SubElement(new_root, "worldbody")
        body = etree.SubElement(worldbody, "body", name="object")
        etree.SubElement(body, "freejoint", name="object_free")

        for geom in geoms:
            new_geom = etree.SubElement(body, "geom", attrib=geom.attrib)

        # 保存新 XML
        out_path = os.path.join(os.path.dirname(obj_dir), f"scale{scale_str}_coacd.xml")
        etree.ElementTree(new_root).write(out_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
        print(f"Generated: {out_path}")
