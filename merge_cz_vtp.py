import os
import glob
import numpy as np

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
T_seed_K = 1500.0
T_hot_K = 1850.0

INFER_DIR = "outputs/arch.fully_connected.layer_size=256,arch.fully_connected.nr_layers=6/train_cz_v1/inferencers"
OUT_FILE = "combined_temperature.vtp"


def find_latest(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def read_polydata(fname):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()
    poly = reader.GetOutput()
    return poly


def get_array_names(poly):
    pd = poly.GetPointData()
    names = []
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        if arr is not None:
            names.append(arr.GetName())
    return names


def get_numpy_array(poly, name):
    arr = poly.GetPointData().GetArray(name)
    if arr is None:
        raise RuntimeError(f"Array '{name}' not found. Available: {get_array_names(poly)}")
    return vtk_to_numpy(arr)


def get_points_numpy(poly):
    pts = poly.GetPoints()
    if pts is None:
        return np.empty((0, 3), dtype=np.float32)
    arr = pts.GetData()
    if arr is None:
        return np.empty((0, 3), dtype=np.float32)
    pts_np = vtk_to_numpy(arr)
    return pts_np.reshape(-1, 3)


def build_polydata(points_xyz, theta, temperature_K, region_id):
    n = len(points_xyz)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points_xyz.astype(np.float32), deep=True))

    verts = vtk.vtkCellArray()
    for i in range(n):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(verts)

    theta_vtk = numpy_to_vtk(theta.astype(np.float32), deep=True)
    theta_vtk.SetName("theta")

    temp_vtk = numpy_to_vtk(temperature_K.astype(np.float32), deep=True)
    temp_vtk.SetName("temperature_K")

    region_vtk = numpy_to_vtk(region_id.astype(np.int32), deep=True)
    region_vtk.SetName("region_id")

    poly.GetPointData().AddArray(theta_vtk)
    poly.GetPointData().AddArray(temp_vtk)
    poly.GetPointData().AddArray(region_vtk)
    poly.GetPointData().SetActiveScalars("temperature_K")

    return poly


def main():
    crystal_file = find_latest(os.path.join(INFER_DIR, "**", "*crystal*.vtp"))
    melt_file = find_latest(os.path.join(INFER_DIR, "**", "*melt*.vtp"))
    crucible_file = find_latest(os.path.join(INFER_DIR, "**", "*crucible*.vtp"))
    insulation_file = find_latest(os.path.join(INFER_DIR, "**", "*insulation*.vtp"))

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
        poly = read_polydata(fname)

        npts = poly.GetNumberOfPoints()
        print(f"  points: {npts}")
        print(f"  arrays: {get_array_names(poly)}")

        if npts == 0:
            print(f"  warning: no points in {fname}")
            continue

        pts = get_points_numpy(poly)
        theta = get_numpy_array(poly, theta_name)

        tempK = T_seed_K + theta * (T_hot_K - T_seed_K)

        all_points.append(pts)
        all_theta.append(theta)
        all_tempK.append(tempK)
        all_region.append(np.full(len(theta), region_id, dtype=np.int32))

    if not all_points:
        raise RuntimeError("No usable inferencer VTP files found.")

    points_xyz = np.vstack(all_points)
    theta = np.concatenate(all_theta)
    temperature_K = np.concatenate(all_tempK)
    region_id = np.concatenate(all_region)

    print("\nCombined summary:")
    print("  total points:", len(points_xyz))
    print("  temperature_K min/max:", float(np.min(temperature_K)), float(np.max(temperature_K)))

    poly_out = build_polydata(points_xyz, theta, temperature_K, region_id)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(OUT_FILE)
    writer.SetInputData(poly_out)
    writer.SetDataModeToBinary()
    writer.Write()

    print(f"\nSaved: {os.path.abspath(OUT_FILE)}")
    print("Open in ParaView and:")
    print("  1. Click Apply")
    print("  2. Reset Camera")
    print("  3. Representation = Points or Point Gaussian")
    print("  4. Color by 'temperature_K'")


if __name__ == "__main__":
    main()
