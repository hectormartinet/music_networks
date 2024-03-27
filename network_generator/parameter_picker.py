import wx
import random

class ParameterPicker(wx.Frame):
    def __init__(self, parent, id, title, widgets_type, default_params, dependancy={}):
        self.width = 500
        self.height = 500
        wx.Frame.__init__(self, parent, id, title, size=(self.width, self.height))
        self.panel = wx.Panel(self, -1)
        self.interval = 27
        self.checkbox = {}
        self.folder_pickers = {}
        self.number_pickers = {}
        ypos = 10
        self.params = default_params
        self.widgets = {}
        for param, value in default_params.items():
            w_type = widgets_type[param]
            if not w_type in self.widgets:
                self.widgets[w_type] = {}
            if w_type == "checkbox":
                self.create_checkbox(param, value, ypos)
            elif w_type == "folder_picker":
                self.create_folder_picker(param, value, ypos)
            elif w_type == "number_picker":
                self.create_number_picker(param, value, ypos)
            else:
                print("Widget not recognized")
                assert(False)
            ypos += self.interval
        for child_param, values_to_unpack in dependancy.items():
            parent_param, value = values_to_unpack
            child_type = widgets_type[child_param]
            parent_type = widgets_type[parent_param]
            def auto_activation(event, 
                child_widget=self.widgets[child_type][child_param], value = value,
                parent_widget=self.widgets[parent_type][parent_param]):
                if parent_widget.GetValue() == value:
                    child_widget.Enable()
                else:
                    child_widget.Disable()
            self.Bind(wx.EVT_CHECKBOX, auto_activation, self.widgets[parent_type][parent_param])
            auto_activation(None)
        self.button = wx.Button(self.panel, -1, "Confirm", (self.width-100, self.height-70))
        self.Bind(wx.EVT_BUTTON, self.save, self.button)
        
        self.Centre()
        self.Show()

    def create_checkbox(self, name, default, ypos):
        self.widgets["checkbox"][name] = wx.CheckBox(self.panel, -1, name, (10, ypos))
        self.widgets["checkbox"][name].SetValue(default)
    
    def create_folder_picker(self, name, default, ypos):
        text1 = wx.StaticText(self.panel, -1, name, (10, ypos))
        size = text1.GetSize().x
        self.widgets["folder_picker"][name] = wx.Button(self.panel, -1, "search", (15+size, ypos-4))
        size += self.widgets["folder_picker"][name].GetSize().x
        text2 = wx.StaticText(self.panel, -1, default, (20+size, ypos))
        def save_folder(event):
            dialog = wx.DirDialog(None, "Choose a directory:",style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
            if dialog.ShowModal() == wx.ID_OK:
                self.params[name] = dialog.GetPath()
                text2.SetLabel(self.params[name])

        self.Bind(wx.EVT_BUTTON, save_folder, self.widgets["folder_picker"][name])
    
    def create_number_picker(self, name, default, ypos):
        centeredLabel = wx.StaticText(self.panel, -1, name, (10,ypos))
        size = centeredLabel.GetSize().x

        mlTextCtrl = wx.TextCtrl(self.panel, -1, str(default), (15+size, ypos-3))
        self.widgets["number_picker"][name] = mlTextCtrl
    
    def save(self, event):
        for key, checkbox in self.widgets["checkbox"].items():
            self.params[key] = checkbox.GetValue()
        for key, number_picker in self.widgets["number_picker"].items():
            self.params[key] = float(number_picker.GetValue())
        self.Close()

#---------------------------------------------------------------------------
def get_parameters(default_params):
    widgets_type = {
        "rest":"checkbox",
        "octave":"checkbox",
        "pitch":"checkbox",
        "duration":"checkbox",
        "offset":"checkbox",
        "offset_period":"number_picker",
        "transpose":"checkbox",
        "strict_link":"checkbox",
        "max_link_time_diff":"number_picker",
        "layer":"checkbox",
        "diatonic_interval":"checkbox",
        "chromatic_interval":"checkbox",
        "midi_folder_or_file":"folder_picker",
        "outfolder":"folder_picker"
    }
    dependancy = {
        "octave": ("pitch", True),
        "offset_period": ("offset", True),
        "strict_link": ("layer",True)
    }
    app = wx.App(0)
    picker = ParameterPicker(None, -1, 'Parameters', widgets_type, default_params, dependancy)
    app.MainLoop()
    return picker.params
