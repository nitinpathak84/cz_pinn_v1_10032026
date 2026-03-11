import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
T_seed_K = 1500.0
T_hot_K = 1850.0

INFER_DIR = "outputs/arch.fully_connected.layer_size=256,arch.fully_connected.nr_layers=6/train_cz_v1/inferencers"
OUT_FILE = "combined_temperature.vtp"


def parse_data_array(data_array):
    text = (data_array.text or "").strip()
    if not text:
        return np.array([])
    if data_array.attrib.get("type", "").startswith("Int"):
        return np.fromstring(text, sep=" ", dtype=np.int32)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def read_vtp_points_and_arrays(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    piece = root.find(".//Piece")
    if piece is None:
        raise RuntimeError(f"Could not find Piece in {filename}")

    points_da = root.find(".//Points/DataArray")
    if points_da is None:
        raise RuntimeError(f"Could not find Points/DataArray in {filename}")

    pts = parse_data_array(points_da).reshape(-1, 3)

    pdata = root.find(".//PointData")
    arrays = {}
    if pdata is not None:
        for da in pdata.findall("DataArray"):
            name = da.attrib.get("Name", "")
            arrays[name] = parse_data_array(da)

    return pts, arrays


def vtk_data_array(parent, name, data, ncomp=1, dtype="Float32"):
    arr = ET.SubElement(
        parent,
        "DataArray",
        type=dtype,
        Name=name,
        NumberOfComponents=str(ncomp),
        format="ascii",
    )
    if ncomp == 1:
        arr.text = "\n" + " ".join(f"{float(v):.8f}" for v in data) + "\n"
    else:
        flat = []
        for row in data:
            flat.extend([f"{float(v):.8f}" for v in row])
        arr.text = "\n" + " ".join(flat) + "\n"
    return arr


def vtk_int_array(parent, name, data):
    arr = ET.SubElement(
        parent,
        "DataArray",
        type="Int32",
        Name=name,
        NumberOfComponents="1",
        format="ascii",
    )
    arr.text = "\n" + " ".join(str(int(v)) for v in data) + "\n"
    return arr


def write_vtp(filename, points_xyz, point_data):
    n = points_xyz.shape[0]

    vtkfile = ET.Element(
        "VTKFile",
        type="PolyData",
        version="0.1",
        byte_order="LittleEndian",
    )
    polydata = ET.SubElement(vtkfile, "PolyData")
    piece = ET.SubElement(
        polydata,
        "Piece",
        NumberOfPoints=str(n),
        NumberOfVerts=str(n),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )

    points = ET.SubElement(piece, "Points")
    vtk_data_array(points, "Points", points_xyz, ncomp=3, dtype="Float32")

    verts = ET.SubElement(piece, "Verts")
    vtk_int_array(verts, "connectivity", np.arange(n, dtype=np.int32))
    vtk_int_array(verts, "offsets", np.arange(1, n + 1, dtype=np.int32))

    pdata = ET.SubElement(piece, "PointData")
    for name, values in point_data.items():
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.integer):
            vtk_int_array(pdata, name, values)
        else:
            vtk_data_array(pdata, name, values, ncomp=1, dtype="Float32")

    tree = ET.ElementTree(vtkfile)
    ET.indent(tree, space="  ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)


def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def main():
    crystal_file = find_latest(os.path.join(INFER_DIR, "*crystal*.vtp"))
    melt_file = find_latest(os.path.join(INFER_DIR, "*melt*.vtp"))
    crucible_file = find_latest(os.path.join(INFER_DIR, "*crucible*.vtp"))
    insulation_file = find_latest(os.path.join(INFER_DIR, "*insulation*.vtp"))

    files = [
        ("crystal", 1, "theta_cr", crystal_file),
        ("melt", 2, "theta_m", melt_file),
        ("crucible", 3, "theta_cu", crucible_file),
        ("insulation", 4, "theta_ins", insulation_file),
    ]

    all_points = []
    all_theta = []
    all_tempK = []
    all_region = []

    for region_name, region_id, theta_name, fname in files:
        if fname is None:
            print(f"Skipping {region_name}: no matching file found")
            continue

        print(f"Reading {region_name}: {fname}")
        pts, arrays = read_vtp_points_and_arrays(fname)

        if theta_name not in arrays:
            raise RuntimeError(
                f"{fname} does not contain expected array '{theta_name}'. "
                f"Available arrays: {list(arrays.keys())}"
            )

        theta = arrays[theta_name]
        tempK = T_seed_K + theta * (T_hot_K - T_seed_K)

        all_points.append(pts)
        all_theta.append(theta)
        all_tempK.append(tempK)
        all_region.append(np.full(len(theta), region_id, dtype=np.int32))

    if not all_points:
        raise RuntimeError("No inferencer VTP files found.")

    points = np.vstack(all_points)
    theta = np.concatenate(all_theta)
    temperature_K = np.concatenate(all_tempK)
    region_id = np.concatenate(all_region)

    write_vtp(
        OUT_FILE,
        points,
        {
            "theta": theta,
            "temperature_K": temperature_K,
            "region_id": region_id,
        },
    )

    print(f"\nSaved combined file: {os.path.abspath(OUT_FILE)}")
    print("Open this file in ParaView and color by 'temperature_K'.")


if __name__ == "__main__":
    main()