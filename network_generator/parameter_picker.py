import wx

class ParameterPicker(wx.Frame):
    def __init__(self, parent, id, title, widgets_type, default_params, dependancy={}, widget_data={}):
        self.interval = 27
        self.width = 500
        self.height = self.interval*(len(default_params)+3)
        wx.Frame.__init__(self, parent, id, title, size=(self.width, self.height))
        self.panel = wx.Panel(self, -1)
        self.checkbox = {}
        self.folder_pickers = {}
        self.number_pickers = {}
        ypos = 10
        self.params = default_params
        self.widgets = {}
        self.widget_data = widget_data
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
            elif w_type == "file_picker":
                self.create_file_picker(param, value, ypos)
            elif w_type == "choice":
                possible_values = widget_data[param]
                self.create_choice(param, possible_values, value, ypos)
            else:
                print("Widget not recognized")
                assert(False)
            ypos += self.interval
        for parent_param, param_and_value_list in dependancy.items():
            parent_type = widgets_type[parent_param]
            def auto_activation(event, lst=param_and_value_list, parent_widget=self.widgets[parent_type][parent_param]):
                for child_param, value in lst:
                    child_type = widgets_type[child_param]
                    child_widget = self.widgets[child_type][child_param]
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
    
    def create_file_picker(self, name, default, ypos):
        text1 = wx.StaticText(self.panel, -1, name, (10, ypos))
        size = text1.GetSize().x
        self.widgets["file_picker"][name] = wx.Button(self.panel, -1, "search", (15+size, ypos-4))
        size += self.widgets["file_picker"][name].GetSize().x
        text2 = wx.StaticText(self.panel, -1, ",".join(default), (20+size, ypos))
        def save_folder(event):
            dialog = wx.FileDialog(None, "Choose a file:",style=wx.DD_DEFAULT_STYLE | wx.FD_MULTIPLE)
            if dialog.ShowModal() == wx.ID_OK:
                self.params[name] = dialog.GetPaths()
                text2.SetLabel(",".join(self.params[name]))
        self.Bind(wx.EVT_BUTTON, save_folder, self.widgets["file_picker"][name])
    
    def create_number_picker(self, name, default, ypos):
        centeredLabel = wx.StaticText(self.panel, -1, name, (10,ypos))
        size = centeredLabel.GetSize().x

        self.widgets["number_picker"][name] = wx.TextCtrl(self.panel, -1, str(default), (15+size, ypos-3))
    
    def create_choice(self, name, values, default, ypos):
        text1 = wx.StaticText(self.panel, -1, name, (10, ypos))
        size = text1.GetSize().x
        self.widgets["choice"][name] = wx.Choice(self.panel, -1, (15+size,ypos), choices = values, name=name)
        n = values.index(default)
        self.widgets["choice"][name].SetSelection(n)

    def save(self, event):
        for key, checkbox in self.widgets["checkbox"].items():
            self.params[key] = checkbox.GetValue()
        for key, number_picker in self.widgets["number_picker"].items():
            self.params[key] = float(number_picker.GetValue())
        for key, choice in self.widgets["choice"].items():
            self.params[key] = self.widget_data[key][choice.GetSelection()]
        self.Close()

#---------------------------------------------------------------------------
def get_parameters(default_params):
    widgets_type = {
        "rest":"checkbox",
        "octave":"checkbox",
        "enharmony":"checkbox",
        "pitch":"checkbox",
        "duration":"checkbox",
        "offset":"checkbox",
        "offset_period":"number_picker",
        "transpose":"checkbox",
        "strict_link":"checkbox",
        "max_link_time_diff":"number_picker",
        "structure":"choice",
        "diatonic_interval":"checkbox",
        "chromatic_interval":"checkbox",
        "chord_function":"checkbox",
        "group_by_beat":"checkbox",
        "split_chords":"checkbox",
        "duration_weighted_intergraph":"checkbox",
        "analyze_key":"checkbox",
        "keep_extra":"checkbox",
        "midi_files":"file_picker",
        "outfolder":"folder_picker"
    }

    widget_data = {
        "structure":["multilayer", "monolayer", "chordify"]
    }

    # parent_parameters:[(child_parameter, required_value),...]
    # Some parameters are only usefull when an other has a specific value
    dependancy = {
        "pitch": [("octave",True),("enharmony",True)],
        "offset": [("offset_period", True)],
        "enharmony":[("diatonic_interval", False)]
    }
    app = wx.App(0)
    picker = ParameterPicker(None, -1, 'Parameters', widgets_type, default_params, dependancy, widget_data)
    app.MainLoop()
    return picker.params
