import wx
import random

class ParameterPicker(wx.Frame):
    def __init__(self, parent, id, title, **kwargs):
        self.width = 500
        self.height = 500
        wx.Frame.__init__(self, parent, id, title, size=(self.width, self.height))
        self.panel = wx.Panel(self, -1)
        self.interval = 27
        self.checkbox = {}
        self.folder_pickers = {}
        self.number_pickers = {}
        ypos = 10
        self.params = kwargs
        for key, value in kwargs.items():
            if type(value) == bool:
                self.create_checkbox(key, value, ypos)
            if type(value) == str:
                self.create_folder_picker(key, value, ypos)
            if type(value) == float:
                self.create_number_picker(key, value, ypos)
            ypos += self.interval
        self.button = wx.Button(self.panel, -1, "Valider", (self.width-100, self.height-70))
        self.Bind(wx.EVT_BUTTON, self.save, self.button)
        
        self.Centre()
        self.Show()

    def create_checkbox(self, name, default, ypos):
        self.checkbox[name] = wx.CheckBox(self.panel, -1, name, (10, ypos))
        self.checkbox[name].SetValue(default)
    
    def create_folder_picker(self, name, default, ypos):
        text1 = wx.StaticText(self.panel, -1, name, (10, ypos))
        size = text1.GetSize().x
        self.folder_pickers[name] = wx.Button(self.panel, -1, "search", (15+size, ypos-4))
        size += self.folder_pickers[name].GetSize().x
        text2 = wx.StaticText(self.panel, -1, default, (20+size, ypos))
        def save_folder(event):
            dialog = wx.DirDialog(None, "Choose a directory:",style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
            if dialog.ShowModal() == wx.ID_OK:
                self.params[name] = dialog.GetPath()
                text2.SetLabel(self.params[name])

        self.Bind(wx.EVT_BUTTON, save_folder, self.folder_pickers[name])
    
    def create_number_picker(self, name, default, ypos):
        centeredLabel = wx.StaticText(self.panel, -1, name, (10,ypos))
        size = centeredLabel.GetSize().x

        mlTextCtrl = wx.TextCtrl(self.panel, -1, str(default), (15+size, ypos-3))
        self.number_pickers[name] = mlTextCtrl
    
    def save(self, event):
        for key, checkbox in self.checkbox.items():
            self.params[key] = checkbox.GetValue()
        for key, number_picker in self.number_pickers.items():
            self.params[key] = float(number_picker.GetValue())
        self.Close()

#---------------------------------------------------------------------------
def get_parameters(**kwargs):
    app = wx.App(0)
    picker = ParameterPicker(None, -1, 'wx.CheckBox',**kwargs)
    app.MainLoop()
    return picker.params
