from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class MasterVolume:
    def __init__(self):
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, 
            CLSCTX_ALL,
            None
        )
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        print('Volume manager ready current volume', self.GetVolume())
    
    def SetVolume(self, value):
        self.volume.SetMasterVolumeLevel(value, None)
    
    def GetVolume(self):
        return self.volume.GetVolumeRange()