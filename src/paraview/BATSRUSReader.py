from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smproxy,
    smproperty,
    smdomain,
    smhint,
)
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject
from vtkmodules.numpy_interface import dataset_adapter as dsa
import numpy as np
import struct
import flekspy


@smproxy.reader(
    name="BATSReader",  # Internal name for the reader
    label="BATSRUS Simulation Reader",  # Name displayed in ParaView's UI
    extensions="out",  # custom file extension(s), space-separated if multiple
    file_description="BATSRUS Files",
)
class BATSReader(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkImageData"
        )
        self._filename = None
        self._data_extents = [0, 0, 0, 0, 0, 0]
        self.dim = 3

        # If data can be time-varying, we'll need these:
        # self._timesteps = []
        # self._current_timestep_index = 0

    @smproperty.stringvector(name="FileName", panel_visibility="advanced")
    @smdomain.filelist()
    @smhint.filechooser(extensions="out", file_description="BATSRUS Files")
    def SetFileName(self, name):
        """Specify the filename for the BATSRUS simulation data."""
        if self._filename != name:
            self._filename = name
            self.Modified()  # Important: Mark the algorithm as modified
            # If the filename is cleared, reset relevant internal state
            if not name:
                self._data_extents = [0, 0, 0, 0, 0, 0]
                # self._timesteps = []

    def _get_file_series(self):
        """
        Helper function if your reader needs to handle file series for time-dependent data.
        For a single file, this might not be strictly necessary, but good for extensibility.
        """
        if not self._filename:
            return []
        # Example: if files are named like sim_000.out, sim_001.out, etc.
        # For a single file per timestep, often this is handled by ParaView's file grouping.
        # If a single file contains multiple timesteps, we'll parse them in RequestInformation.
        return [self._filename]  # Simplified for a single file

    # --- RequestInformation: Provide metadata about the data ---
    def RequestInformation(self, request, inInfoVec, outInfoVec):
        if not self._filename:
            print("BATSReader: No filename set.")
            return 0  # Indicate failure

        print(f"BATSReader: RequestInformation for {self._filename}")

        # (1) Use flekspy to get metadata (e.g., dimensions) from your file
        try:
            with open(self._filename, "rb") as f:
                end_char = "<"  # Endian marker (default: little)
                record_len_raw = f.read(4)
                record_len = (struct.unpack(end_char + "l", record_len_raw))[0]
                f.read(record_len)  # skip headline

                (old_len, record_len) = struct.unpack(end_char + "2l", f.read(8))
                pformat = "f"
                # Parse rest of header; detect double-precision file
                if record_len > 20:
                    pformat = "d"
                (iter, time, ndim, nparam, nvar) = struct.unpack(
                    "{0}l{1}3l".format(end_char, pformat), f.read(record_len)
                )
                gencoord = ndim < 0
                self.ndim = abs(ndim)
                # Get gridsize
                (old_len, record_len) = struct.unpack(end_char + "2l", f.read(8))
                grid = np.array(
                    struct.unpack(
                        "{0}{1}l".format(end_char, self.ndim), f.read(record_len)
                    )
                )

            if self.ndim == 3:
                nx, ny, nz = grid
            elif self.ndim == 2:
                nx, ny = grid
                nz = 1
            elif self.ndim == 1:
                nx = grid
                ny, nz = 1, 1
            self._data_extents = [0, nx - 1, 0, ny - 1, 0, nz - 1]
            print(f"BATSReader: Determined extents: {self._data_extents}")

        except Exception as e:
            print(f"BATSReader: Error getting metadata: {e}")
            return 0

        # (2) Set the WHOLE_EXTENT for vtkImageData
        # This tells ParaView the overall dimensions of the structured grid.
        output_info = outInfoVec.GetInformationObject(0)
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

        output_info.Set(
            vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(), self._data_extents, 6
        )

        # (3) Handle Time Steps (if applicable)
        # If your file format supports multiple time steps:
        #   timesteps_from_file = flekspy.get_timesteps(self._filename)
        #   if timesteps_from_file:
        #       self._timesteps = list(timesteps_from_file)
        #       output_info.Set(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), self._timesteps, len(self._timesteps))
        #       timeRange = [self._timesteps[0], self._timesteps[-1]]
        #       output_info.Set(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), timeRange, 2)
        #   else: # If no time steps or single time step
        #       output_info.Remove(vtkStreamingDemandDrivenPipeline.TIME_STEPS())
        #       output_info.Remove(vtkStreamingDemandDrivenPipeline.TIME_RANGE())

        # (4) Specify Data Array Information (optional but good for performance)

        return 1  # Indicate success

    # --- RequestData: Load the actual data ---
    def RequestData(self, request, inInfoVec, outInfoVec):
        if not self._filename:
            print("BATSReader: No filename set for RequestData.")
            return 0

        print(f"BATSReader: RequestData for {self._filename}")
        output_port_info = outInfoVec.GetInformationObject(0)
        output_data = vtkImageData.GetData(output_port_info)

        # (1) Get the requested update extent
        # For a simple reader loading the whole dataset, this might be the WHOLE_EXTENT.
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

        update_extents = output_port_info.Get(
            vtkStreamingDemandDrivenPipeline.UPDATE_EXTENT()
        )

        # (2) Handle Time (if applicable)
        # current_time_step_value = 0.0
        # if output_port_info.Has(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()):
        #    current_time_step_value = output_port_info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
        #    # self._current_timestep_index = self._timesteps.index(current_time_step_value)

        # (3) Load data using flekspy
        try:
            ds = flekspy.load(self._filename)

        except Exception as e:
            print(f"BATSReader: Error loading data with flekspy: {e}")
            return 0

        # (4) Convert data to vtkImageData
        output_data.SetExtent(*self._data_extents)

        # Set Spacing
        dx = ds.data["x"][1][0] - ds.data["x"][0][0]
        dy = ds.data["y"][0][1] - ds.data["y"][0][0]
        dz = 1.0
        output_data.SetSpacing(dx, dy, dz)

        # Set Origin
        ox = ds.data["x"][0][0]
        oy = ds.data["y"][0][0]
        oz = 0.0
        output_data.SetOrigin(ox, oy, oz)

        # Add the actual data array to the vtkImageData
        num_components = 1  # For scalar data

        for i, name in enumerate(ds.data.name):
            if name.endswith("x") and name != "x":
                num_components = 3
                sim_data = ds.data.array[i : i + 3]
                vtk_data_array = dsa.numpyTovtkDataArray(
                    sim_data.ravel(order="F"), name=name[:-1]
                )
            elif name.startswith("ux"):
                num_components = 3
                sim_data = ds.data.array[i : i + 3]
                vtk_data_array = dsa.numpyTovtkDataArray(
                    sim_data.ravel(order="F"), name="u" + name[-2:]
                )
            elif name.startswith("pXX"):
                num_components = 6
                sim_data = ds.data.array[i : i + 6]
                vtk_data_array = dsa.numpyTovtkDataArray(
                    sim_data.ravel(order="F"), name="pTensor" + name[-2:]
                )
            elif (
                (name.endswith("y") and name != "y")
                or (name.endswith("z") and name != "z")
                or name.startswith("uy")
                or name.startswith("uz")
                or (
                    name.startswith("pY")
                    or name.startswith("pZ")
                    or name.startswith("pX")
                )
            ):
                continue
            else:
                num_components = 1
                sim_data = ds.data.array[i]
                vtk_data_array = dsa.numpyTovtkDataArray(
                    sim_data.ravel(order="F"), name=name
                )
            vtk_data_array.SetNumberOfComponents(num_components)
            output_data.GetPointData().AddArray(vtk_data_array)

        # for key, sim_data in data_dict.items():
        # vtk_data_array = dsa.numpyTovtkDataArray(
        # sim_data.ravel(order="F"), name=key
        # )
        # vtk_data_array.SetNumberOfComponents(num_components)
        #
        # output_data.GetPointData().AddArray(vtk_data_array)

        # (5) Handle Time (if applicable)
        # if self._timesteps:
        #    output_data.GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), current_time_step_value)

        print(f"BATSReader: Successfully prepared vtkImageData for {self._filename}")
        return 1  # Indicate success
