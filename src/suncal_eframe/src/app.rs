// Measurement Quality Assurance Application
use eframe::egui;
use eframe::egui::text::{CCursor, CCursorRange};
use egui_plot::{BarChart, Points, Line, VLine, Bar, Plot, PlotPoints, LineStyle};
use ecolor::Color32;

use sunlib;
use units;
use sunlib::cfg::{MeasureSystem, ModelQuantity, ModelFunction, CorrelationCoeff,
                  ModelCurve, CurveModel, TypeBDist, TypeBTriangular, TypeBUniform, TypeBTolerance};
use sunlib::result::{Histogram, CurveResult};
use sunlib::dists::{linspace, norm_pdf};
use sunlib::curves::curve_eqn;

struct QuantityUi {
    units: String,
    typea: String,  // Type A String data
    stddev: f64,  // Combined stddev for the Quantity
    typebs: Vec<TypeBUi>,
}
impl QuantityUi {
    fn update_model(&mut self, item: &mut ModelQuantity) {
        item.units = Some(self.units.clone());

        // Remove deleted TypeB's
        self.typebs = self.typebs.clone().into_iter().filter(|x| x.dist != TypeBDistUi::DeleteMe).collect::<Vec<TypeBUi>>();

        // Push TypeBs to the ModelQuantity
        item.typeb.clear();
        for typeb in &self.typebs {
            let newb = match &typeb.dist {
                TypeBDistUi::Normal => {
                    TypeBDist::Tolerance(TypeBTolerance{
                        tolerance: typeb.param,
                        confidence: typeb.conf,
                        degf: typeb.degf,
                        kfactor: typeb.kfactor,
                        name: typeb.name.clone(),
                    })
                },
                TypeBDistUi::Uniform => {
                    TypeBDist::Uniform(TypeBUniform{
                        a: typeb.param,
                        degf: typeb.degf,
                        name: typeb.name.clone(),
                    })
                },
                TypeBDistUi::Triangular => {
                    TypeBDist::Triangular(TypeBTriangular{
                        a: typeb.param,
                        degf: typeb.degf,
                        name: typeb.name.clone(),
                    })
                },
                _ => unreachable!(),
            };
            item.typeb.push(newb);
        }

        // Parse Type A string
        let values = self.typea.split('\n').collect::<Vec<&str>>();
        if values.len() > 0 {
            if values[0].chars().any(|x| x==',') {
                // Has comma, parse reproducibility
                let mut reprod_values: Vec<Vec<f64>> = Vec::new();
                for row in values.iter() {
                    let rvalues = row.split(',').collect::<Vec<&str>>();
                    let floats = rvalues.iter().map(|x| x.trim_ascii_start().trim_ascii_end().parse::<f64>().unwrap_or(f64::NAN)).filter(|x| x.is_finite()).collect::<Vec<f64>>();
                    if floats.len() > 0 {
                        reprod_values.push(floats);
                    };
                }
                item.reproducibility = match reprod_values.len() {
                    0 => None,
                    _ => Some(reprod_values),
                };
                item.repeatability = None;
            } else {
                // Repeatability (one value per line)
                let floats = values.iter().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).filter(|x| x.is_finite()).collect::<Vec<f64>>();
                item.repeatability = match floats.len() {
                    0 => None,
                    _ => Some(floats),
                };
                item.reproducibility = None;
            }
        
        } else {
            item.repeatability = None;
            item.reproducibility = None;
        }

        self.stddev = item.variance(&Vec::new()).sqrt();
    }
}


struct CurveUi {
    data: String,  // XY data as string
    guess: Vec<f64>,
}
impl CurveUi {
    fn new() -> Self {
        Self{
            data: String::new(),
            guess: vec![1.0; 4],
        }
    }
    fn update_model(&mut self, item: &mut ModelCurve) {
        // parse data to x, y
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        let mut uy: Vec<f64> = Vec::new();

        for row in self.data.lines() {
            let values = row.split(&[',', ';']).collect::<Vec<&str>>();
            if values.len() == 2 {
                x.push(values[0].trim_ascii_start().trim_ascii_end().parse::<f64>().unwrap_or(0.0));
                y.push(values[1].trim_ascii_start().trim_ascii_end().parse::<f64>().unwrap_or(0.0));
            } else if values.len() == 3 {
                x.push(values[0].trim_ascii_start().trim_ascii_end().parse::<f64>().unwrap_or(0.0));
                y.push(values[1].trim_ascii_start().trim_ascii_end().parse::<f64>().unwrap_or(0.0));
                uy.push(values[2].trim_ascii_start().trim_ascii_end().parse::<f64>().unwrap_or(0.0));
            }
        }
        item.x = x;
        item.y = y;
        item.uy = if uy.len() > 0 { Some(uy) } else { None };
        item.guess = Some(self.guess.clone());
    }
}



#[derive(Clone, Debug, PartialEq)]
enum Ktype {
    K,
    Confidence
}

#[derive(Clone, Debug, PartialEq)]
enum TypeBDistUi {
    Normal,
    Uniform,
    Triangular,
    DeleteMe,
}
#[derive(Clone, Debug)]
struct TypeBUi {
    dist: TypeBDistUi,
    param: f64,  // stdev or a
    ktype: Ktype,
    kfactor: f64,
    conf: f64,
    degf: f64,
    name: String,
}


pub struct SuncalApp {
    model: MeasureSystem,
    quantities: Vec<QuantityUi>,
    curves: Vec<CurveUi>,
    save_window_open: bool,
    load_window_open: bool,
    show_output: bool,
    config_to_load: String,
    load_message: String,
    nsamples: usize,
    decimals: usize,

    varnames: Vec<String>, // List of quantity symbols
    output_raw: String,  // Raw output string (like CLI output)
    output_summary: String, //
    output_var: String,  // Currently displayed output quantity
    output_reports: Vec<String>,  // Reports for all the varnames
    output_hists: Vec<Option<Histogram>>,  // Histograms of all the MC variables
    output_gum_means: Vec<f64>,  // GUM expected values
    output_gum_stds: Vec<f64>,    // GUM expanded stdev
    output_gum_k: Vec<f64>,    // GUM expanded k
    output_curves: Vec<Option<CurveResult>>,
}
impl Default for SuncalApp {
    fn default() -> Self {
        let mut def = Self {
            model: MeasureSystem::default(),
            quantities: vec![],
            curves: vec![],
            save_window_open: false,
            load_window_open: false,
            show_output: true,
            config_to_load: String::new(),
            load_message: String::new(),
            nsamples: 100000,
            decimals: 4,

            varnames: Vec::new(),
            output_raw: String::new(),
            output_summary: String::new(),
            output_var: String::from("Summary"),
            output_reports: Vec::new(),
            output_hists: Vec::new(),
            output_gum_means: Vec::new(),
            output_gum_stds: Vec::new(),
            output_gum_k: Vec::new(),
            output_curves: Vec::new(),
        };
        def.recalc();  // Recalc all or 0 - or not needed
        def
    }
}
impl SuncalApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        units::init();
        Default::default()
    }
    fn recalc(&mut self) {
        if self.quantities.len() > 0 {
            for i in 0..self.quantities.len() {
                self.quantities[i].update_model(&mut self.model.quantity[i]);
            }
        }
        if self.curves.len() > 0 {
            for i in 0..self.curves.len() {
                self.curves[i].update_model(&mut self.model.curve[i]);
            }
        }

        // Calculate everything
        let result = self.model.calculate();
        match result {
            Ok(r) => {
                self.output_reports.clear();
                self.output_hists.clear();
                self.output_curves.clear();
                self.varnames = r.varnames();

                self.output_summary = r.summary(self.decimals);
                self.output_raw = r.to_string();
                (self.output_gum_means, self.output_gum_stds, self.output_gum_k) = r.gum_expected();

                for varname in &self.varnames {
                    self.output_reports.push(
                        r.get_quantity(&varname, self.decimals)
                    );
                    self.output_hists.push(
                        r.get_histogram(&varname)
                    );
                    self.output_curves.push(
                        r.get_curve(&varname)
                    );
                };
            },
            Err(e) => {
                self.output_reports.clear();
                self.output_summary = e.to_string();
                self.output_raw = e.to_string();
            },
        };
    }
    fn calc_montecarlo(&mut self) {
        self.model.settings.montecarlo = self.nsamples;
        self.recalc();
        self.model.settings.montecarlo = 0;
    }

    fn load_model(&mut self, model: MeasureSystem) {
        self.model = model;
        self.model.settings.montecarlo = 0;
        self.quantities.clear();
        self.curves.clear();

        for qty in &self.model.quantity {
            let units = match &qty.units {
                Some(u) => u.clone(),
                None => String::new(),
            };

            let mut typebs: Vec<TypeBUi> = Vec::new();
            for typeb in &qty.typeb {
                let newb = match &typeb {
                    TypeBDist::Tolerance(b) => {
                        TypeBUi{
                            dist: TypeBDistUi::Normal,
                            param: b.tolerance,
                            degf: b.degf,
                            conf: b.confidence,
                            kfactor: b.kfactor,
                            ktype: if b.kfactor.is_finite() {Ktype::K} else {Ktype::Confidence},
                            name: b.name.clone(),
                        }
                    },
                    TypeBDist::Uniform(b) => {
                        TypeBUi{
                            dist: TypeBDistUi::Uniform,
                            param: b.a,
                            degf: b.degf,
                            conf: f64::NAN,
                            kfactor: f64::NAN,
                            ktype: Ktype::Confidence,
                            name: b.name.clone(),
                        }
                    },
                    TypeBDist::Triangular(b) => {
                        TypeBUi{
                            dist: TypeBDistUi::Triangular,
                            param: b.a,
                            degf: b.degf,
                            conf: f64::NAN,
                            kfactor: f64::NAN,
                            ktype: Ktype::Confidence,
                            name: b.name.clone(),
                        }
                    },
                    TypeBDist::Normal(b) => {
                        TypeBUi{
                            dist: TypeBDistUi::Normal,
                            param: b.stddev,
                            conf: 0.6827,
                            degf: b.degf,
                            kfactor: f64::NAN,
                            ktype: Ktype::Confidence,
                            name: b.name.clone(),
                        }
                    }
                    _ => unreachable!(),
                };
                typebs.push(newb);
            }

            self.quantities.push(
                QuantityUi{
                    units: units,
                    stddev: qty.variance(&Vec::new()).sqrt(),
                    typebs: typebs,
                    typea: String::new(),
                }
            );
        }

        for curve in &self.model.curve {
            let guess: Vec<f64> = match &curve.guess {
                Some(g) => g.clone(),
                None => vec![1.0; 4],
            };
            self.curves.push(
                CurveUi{
                    data: curve.data_string(),
                    guess: guess,
                }
            );
        }

    }
   fn draw_menu(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            let response = ui.button("ℹ").on_hover_text("About");
            let popup_id = ui.make_persistent_id("about");
            if response.clicked() {
                ui.memory_mut(|mem| mem.toggle_popup(popup_id));
            }
            egui::popup::popup_above_or_below_widget(
                ui, popup_id, &response,
                egui::AboveOrBelow::Below,
                egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                    ui.set_min_width(400.0);
                    ui.heading("Suncal - Sandia Uncertainty Calculator");
                    ui.add_space(20.0);
                    ui.strong("Primary Standards Laboratory");
                    ui.strong("Sandia National Laboratories");
                    ui.add_space(20.0);
                    ui.label("© 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.");
                    ui.add_space(20.0);
                    ui.label("This software is distributed under the GNU General Public License.");
                    ui.add_space(20.0);
                    ui.hyperlink_to("Homepage", "https://sandialabs.github.io/suncal/suncal/index.html");
            });

            if ui.button("💾")
                .on_hover_text("Export configuration")
                .clicked() {
                    self.save_window_open = true;
            }
            if ui.button("🗁")
                .on_hover_text("Import configuration")
                .clicked() {
                    self.load_message = String::new();
                    self.load_window_open = true;
            }
            let response = ui.button("⛭").on_hover_text("Configuration");
            let popup_id = ui.make_persistent_id("config");
            if response.clicked() {
                ui.memory_mut(|mem| mem.toggle_popup(popup_id));
            }
            egui::popup::popup_above_or_below_widget(
                ui, popup_id, &response,
                egui::AboveOrBelow::Below,
                egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                    ui.set_min_width(180.0);
                    ui.horizontal(|ui| {
                        ui.label("Level of Confidence");
                        if ui.add(egui::DragValue::new(&mut self.model.settings.confidence)
                        .speed(0.001).range(0.01..=0.9999)
                        .suffix("%")
                        .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                        .custom_parser(|s| {
                            let p = s.parse::<f64>();
                            match p { Ok(v) => Some(v / 100.0), _ => None,}
                        }))
                        .on_hover_text("Level of confidence")
                        .changed() {
                            self.recalc();
                        };
                    });
                    ui.horizontal(|ui| {
                        ui.label("Monte Carlo Samples");
                        ui.add(egui::DragValue::new(&mut self.nsamples));

                    });
                    ui.horizontal(|ui| {
                        ui.label("Decimal Places");
                        if ui.add(egui::DragValue::new(&mut self.decimals).range(0..=12)).changed() {self.recalc()};
                    });

                });

            ui.add_space(40.0);
            ui.separator();
            if ui.button("📏 Quantity").on_hover_text("Add Quantity").clicked() {
                let newq = ModelQuantity::new();
                let units = match &newq.units {
                    Some(u) => u.clone(),
                    None => String::new(),
                };
                self.quantities.push(
                    QuantityUi{
                        units: units,
                        stddev: newq.variance(&Vec::new()).sqrt(),
                        typebs: Vec::new(),
                        typea: String::new(),
                    }
                );
                self.model.quantity.push(
                    newq
                );
                self.recalc();
            };
            if ui.button("🖩 Function").on_hover_text("Add Function").clicked() {
                let newf = ModelFunction::new();
                self.model.function.push(newf);
                self.recalc();
            };
            if ui.button("🗠 Curve").on_hover_text("Add Curve Fit").clicked() {
                let newc = ModelCurve::new(CurveModel::Line, Vec::new(), Vec::new());
                self.model.curve.push(newc);
                self.curves.push(CurveUi::new());
                self.recalc();
            };
            ui.separator();
            ui.add_space(40.0);
            ui.toggle_value(&mut self.show_output, "🖹 Results").on_hover_text("Show raw output");
            ui.add_space(40.0);
            egui::widgets::global_theme_preference_buttons(ui);
        });
        if self.save_window_open || self.load_window_open {
            ui.disable();
        }
    }
    fn draw_save_window(&mut self, ctx: &egui::Context ) {
        // Draw the "Save" dialog
        let mut cfg = match self.model.get_config() {
            Ok(m) => m,
            Err(e) => e.to_string(),
        };

        egui::Window::new("Copy Configuration")
        .open(&mut self.save_window_open)
        .fade_in(true).fade_out(true)
        .show(ctx, |ui| {
            ui.label("Copy (Ctrl+C) this configuration data and save it to a file. It can be pasted back in using the Import window.");
            let mut text = egui::TextEdit::multiline(&mut cfg)
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .show(ui);
            text.state.cursor.set_char_range(Some(
                CCursorRange::two(
                CCursor::new(0), 
                CCursor::new(cfg.len()))
            ));
            text.state.store(ui.ctx(), text.response.id);
            text.response.request_focus();
        });
    }
    fn draw_load_window(&mut self, ctx: &egui::Context ) {
        // Draw the "Load" dialog
        let load_msg = self.load_message.clone();
        let mut closeme = false;
        let mut newmodel: Option<MeasureSystem> = None;

        egui::Window::new("Paste Configuration")
        .open(&mut self.load_window_open)
        .min_width(400.0).min_height(600.0)
        .fade_in(true).fade_out(true)
        .show(ctx, |ui| {
            if ui.button("Import").clicked() {
                let m = MeasureSystem::load_toml(&self.config_to_load);
                match m {
                    Ok(v) => { newmodel = Some(v); closeme = true; },
            Err(e) => {self.load_message = e.to_string()},
                }
            }
            ui.label("Paste a saved configuration here or select example to import.");
            ui.label(egui::RichText::new(load_msg).color(egui::Color32::from_rgb(255, 0, 0)));
            egui::ScrollArea::vertical()
                .id_salt("editor")
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.add(egui::TextEdit::multiline(&mut self.config_to_load)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                        .desired_rows(20)
                    );
            });
        });
        if closeme {
            self.load_window_open = false;
            self.load_model(newmodel.unwrap());
            self.recalc();  // ALL
        }
    }

    fn draw_qty_table(&mut self, ui: &mut egui::Ui) {
        // Direct Quantities
        ui.heading("Measured Quantities");
        let mut need_recalc = false;
        egui::Grid::new("qty_grid").show(ui, |ui| {
            ui.label("");
            ui.strong("Symbol");
            ui.strong("Value").on_hover_text("Measured or expected value");
            ui.strong("Std. Uncertainty").on_hover_text("Standard (k=1) Uncertainty");
            ui.strong("Units").on_hover_text("Units of measure, or leave blank for dimensionless");
            ui.strong("Description").on_hover_text("Descriptive name of the quantity");
            ui.end_row();

            for idx in 0..self.model.quantity.len() {
                let item = &mut self.model.quantity[idx];
                if ui.button("⊟").on_hover_text("Remove quantity").clicked() {
                    self.model.quantity.remove(idx);
                    self.quantities.remove(idx);
                    need_recalc = true;
                    break;
                }
                if ui.add(egui::TextEdit::singleline(&mut item.symbol).min_size(egui::Vec2::new(80.0, 0.0))).changed() {need_recalc=true;}

                if let Some(_) = &item.repeatability {
                    ui.label(format!{"{:.4}", item.expected()});
                } else if let Some(_) = &item.reproducibility {
                    ui.label(format!{"{:.4}", item.expected()});
                } else {
                    if ui.add(egui::DragValue::new(&mut item.measured).speed(0.01)).changed() {need_recalc = true;};
                };

                ui.label(format!("± {:.4}", self.quantities[idx].stddev));
                if ui.add(egui::TextEdit::singleline(&mut self.quantities[idx].units).min_size(egui::Vec2::new(100.0, 0.0))).changed() {need_recalc=true;}
                if ui.add(egui::TextEdit::singleline(&mut item.name).min_size(egui::Vec2::new(100.0, 0.0))).changed() {need_recalc=true;}

                // Type A Collapser
                egui::CollapsingHeader::new("Type A").id_salt(format!("typea_popup{}", idx))
                    .show(ui, |ui| {
                    ui.add(egui::Label::new("Type A Data:                                        ")
                        .wrap_mode(egui::TextWrapMode::Extend))
                        .on_hover_text("Enter repeatability measurements, one value per line\nor reproducibility measurements, separated by commas,\none group per line");
                    if egui::TextEdit::multiline(&mut self.quantities[idx].typea)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                        .show(ui)
                    .response.changed() {
                        need_recalc=true;
                    };
                });
                // Type B Collapser
                egui::CollapsingHeader::new("Type B").id_salt(format!("typeb_popup{}", idx))
                    .show(ui, |ui| {
                        if ui.button("⊞").on_hover_text("Add Type B Uncertainty").clicked() {
                            self.quantities[idx].typebs.push(
                                TypeBUi{
                                    dist: TypeBDistUi::Normal,
                                    param: 1.0,
                                    degf: f64::INFINITY,
                                    conf: 0.95,
                                    kfactor: f64::NAN,
                                    ktype: Ktype::Confidence,
                                    name: "Type B".to_string(),
                                }
                            );
                            need_recalc = true;
                        };
                        for typeb in &mut self.quantities[idx].typebs {
                            ui.horizontal(|ui| {
                                ui.add(egui::TextEdit::singleline(&mut typeb.name).desired_width(50.0)).on_hover_text("Name of uncertainty component");
                                egui::ComboBox::new(ui.next_auto_id(), "")
                                .selected_text(format!("{:?}", typeb.dist))
                                .show_ui(ui, |ui| {
                                    if ui.selectable_value(&mut typeb.dist, TypeBDistUi::Normal, "Normal").clicked() { need_recalc = true; };
                                    if ui.selectable_value(&mut typeb.dist, TypeBDistUi::Uniform, "Uniform").clicked() { need_recalc = true; };
                                    if ui.selectable_value(&mut typeb.dist, TypeBDistUi::Triangular, "Triangular").clicked() { need_recalc = true; };
                                });

                                match &typeb.dist {
                                    TypeBDistUi::Normal => {
                                        ui.label("Uncertainy:");
                                        if ui.add(egui::DragValue::new(&mut typeb.param).speed(0.01)).on_hover_text("Uncertainty").changed() {
                                            need_recalc = true;
                                        };
                                        egui::ComboBox::new(ui.next_auto_id(), "")
                                            .selected_text(format!("{:?}:", typeb.ktype))
                                            .show_ui(ui, |ui| {
                                                if ui.selectable_value(&mut typeb.ktype, Ktype::K, "K-Factor").clicked() {need_recalc=true; typeb.kfactor=2.0};
                                                if ui.selectable_value(&mut typeb.ktype, Ktype::Confidence, "Confidence").clicked() {need_recalc=true; typeb.kfactor=f64::NAN};
                                            });

                                        match typeb.ktype {
                                            Ktype::Confidence => {
                                                if ui.add(egui::DragValue::new(&mut typeb.conf)
                                                .speed(0.001).range(0.01..=0.9999)
                                                .suffix("%")
                                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                .custom_parser(|s| {
                                                    let p = s.parse::<f64>();
                                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                }))
                                                .on_hover_text("Level of confidence")
                                                .changed() {
                                                    need_recalc = true;
                                                };
                                            },
                                            Ktype::K => {
                                                if ui.add(egui::DragValue::new(&mut typeb.kfactor).speed(0.01)).on_hover_text("Coverage factor").changed() {need_recalc=true;}
                                            },
                                        };
                                    },
                                    TypeBDistUi::Uniform => {
                                        ui.label("a");
                                        if ui.add(egui::DragValue::new(&mut typeb.param).speed(0.01)).on_hover_text("Half-width").changed() {
                                            need_recalc = true;
                                        };
                                    },
                                    TypeBDistUi::Triangular => {
                                        ui.label("a");
                                        if ui.add(egui::DragValue::new(&mut typeb.param).speed(0.01)).on_hover_text("Half-width").changed() {
                                            need_recalc = true;
                                        };
                                    },
                                    _ => unreachable!(),
                                }
                                ui.label("ν:");
                                if ui.add(egui::DragValue::new(&mut typeb.degf).speed(0.1)).on_hover_text("Degrees of Freedom").changed() {
                                    need_recalc = true;
                                };
                                if ui.button("⊟").on_hover_text("Remove Type B Component").clicked() {
                                    typeb.dist = TypeBDistUi::DeleteMe;  // Mark it for deletion
                                    need_recalc = true;
                                }
                            });
                        }

                    });
                    ui.end_row();
            }   // Row
        });  // Grid
        
        if self.model.quantity.len() > 1 {
            let mut remove_idx: Option<usize> = None;
            let qty_names: Vec<String> = self.model.quantity.iter().map(|q| q.symbol.clone()).collect();
    
            egui::CollapsingHeader::new("Correlations").id_salt("correlations")
                .show(ui, |ui| {
                    if ui.button("⊞").on_hover_text("Add correlation coefficient").clicked() {
                        let v1 = self.model.quantity[0].symbol.clone();
                        let v2 = self.model.quantity[1].symbol.clone();
                        self.model.correlation.push(
                            CorrelationCoeff{v1: v1, v2: v2, coeff: 0.5}
                        );
                    };

                    for idx in 0..self.model.correlation.len() {
                        let corr = &mut self.model.correlation[idx];
                        ui.horizontal(|ui| {
                            egui::ComboBox::new(ui.next_auto_id(), "")
                              .selected_text(corr.v1.clone())
                              .show_ui(ui, |ui| {
                                for varname in &qty_names {
                                    if ui.selectable_value(&mut corr.v1, varname.to_string(), varname).clicked() { need_recalc = true; };
                                }
                              });
                              ui.label("↔");
                              egui::ComboBox::new(ui.next_auto_id(), "")
                              .selected_text(corr.v2.clone())
                              .show_ui(ui, |ui| {
                                for varname in &qty_names {
                                    if ui.selectable_value(&mut corr.v2, varname.to_string(), varname).clicked() { need_recalc = true; };
                                }
                              });
                        
                              if ui.add(egui::DragValue::new(&mut corr.coeff)
                                .speed(0.001).range(-1.0..=1.0))
                                .on_hover_text("Correlation Coefficient")
                                .changed() {
                                    need_recalc = true;
                                };

                              if ui.button("⊟").on_hover_text("Remove correlation coefficient").clicked() {
                                remove_idx = Some(idx);
                              };
                            });
                    };

                });
            if let Some(idx) = remove_idx {
                self.model.correlation.remove(idx);
                need_recalc = true;
            }
        }
        
        if need_recalc {
            self.recalc();
        }
    }

    fn draw_curve_table(&mut self, ui: &mut egui::Ui) {
        let mut need_recalc = false;
        ui.heading("Curve Fit Quantities");
        egui::Grid::new("curve_grid").show(ui, |ui| {
            ui.label("");
            ui.strong("Model");
            ui.label("");
            ui.strong("Data");
            ui.strong("Guess");
            ui.end_row();
            for idx in 0..self.model.curve.len() {
                let item = &mut self.model.curve[idx];
                if ui.button("⊟").on_hover_text("Remove curve").clicked() {
                    self.model.curve.remove(idx);
                    need_recalc = true;
                    break;
                };
                egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{:?}", item.model))
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut item.model, CurveModel::Line, "Line").clicked() { need_recalc = true; };
                    if ui.selectable_value(&mut item.model, CurveModel::Quadratic, "Quadratic").clicked() { need_recalc = true; };
                    if ui.selectable_value(&mut item.model, CurveModel::Cubic, "Cubic").clicked() { need_recalc = true; };
                    if ui.selectable_value(&mut item.model, CurveModel::Exponential, "Exponential").clicked() { need_recalc = true; };
                    if ui.selectable_value(&mut item.model, CurveModel::DampedSine, "Dampled Sine").clicked() { need_recalc = true; };
                });
                ui.label(curve_eqn(&item.model));

                let response = ui.button("⏷").on_hover_text("Enter measured curve data");
                let popup_id = ui.make_persistent_id("curve");
                if response.clicked() {
                    ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                }
                egui::popup::popup_above_or_below_widget(
                    ui, popup_id, &response,
                    egui::AboveOrBelow::Below,
                    egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                        ui.set_min_width(280.0);
                        ui.label("Enter x, y [, uy] data, comma-separated, one pair per line.");
                        if ui.add(egui::TextEdit::multiline(&mut self.curves[idx].data)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                        ).changed() {need_recalc=true;}
                    });

                    let response = ui.button("⏷").on_hover_text("Initial guess");
                    let popup_id = ui.make_persistent_id("curve_guess");
                    if response.clicked() {
                        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                    }
                    egui::popup::popup_above_or_below_widget(
                        ui, popup_id, &response,
                        egui::AboveOrBelow::Below,
                        egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                            ui.set_min_width(280.0);
                            let n: usize = match item.model {
                                CurveModel::Line => 2,
                                CurveModel::Quadratic => 3,
                                CurveModel::Cubic => 4,
                                CurveModel::Exponential => 2,
                                CurveModel::DampedSine => 3,
                            };
                            for i in 0..n {
                                if ui.add(egui::DragValue::new(&mut self.curves[idx].guess[i]).speed(0.001)).changed() {need_recalc=true;}
                            }

                        });
    

                }
            if need_recalc {
                self.recalc();
            }
        });
    }

    fn draw_func_table(&mut self, ui: &mut egui::Ui) {
        // Functions (Indirect Quantities)
        let mut need_recalc = false;
        ui.heading("Measurement Equations");
        egui::Grid::new("func_grid").show(ui, |ui| {
            ui.label("");
            ui.strong("Symbol");
            ui.strong("Equation").on_hover_text("Calculation as function of Direct Quantities");
            ui.strong("Units").on_hover_text("Units of measure, or leave blank for dimensionless");
            ui.strong("Description").on_hover_text("Description");
            ui.end_row();

            for idx in 0..self.model.function.len() {
                let item = &mut self.model.function[idx];
                if ui.button("⊟").on_hover_text("Remove function").clicked() {
                    self.model.function.remove(idx);
                    need_recalc = true;
                    break;
                };
                if ui.add(egui::TextEdit::singleline(&mut item.symbol).min_size(egui::Vec2::new(80.0, 0.0))).changed() {need_recalc=true;}
                if ui.add(egui::TextEdit::singleline(&mut item.expr).min_size(egui::Vec2::new(260.0, 0.0))).changed() {need_recalc=true;}
                if let Some(ref mut units) = item.units {
                    if ui.add(egui::TextEdit::singleline(units).min_size(egui::Vec2::new(80.0, 0.0))).changed() {need_recalc=true;}
                } else {
                    item.units = Some(String::new());
                    if ui.add(egui::TextEdit::singleline(&mut String::new()).min_size(egui::Vec2::new(80.0, 0.0))).changed() {need_recalc=true;}
                };
                if ui.add(egui::TextEdit::singleline(&mut item.name).min_size(egui::Vec2::new(260.0, 0.0))).changed() {need_recalc=true;}
                ui.end_row();
            }
        });

        if need_recalc {
            self.recalc();
        }
    }

    fn draw_output(&mut self, ctx: &egui::Context) {
        egui::Window::new("Results")
            .show(ctx, |ui| {
                egui::CentralPanel::default()
                .show_inside(ui, |ui| {
    
                ui.horizontal(|ui| {
                    egui::ComboBox::new(ui.next_auto_id(), "")
                        .selected_text(self.output_var.clone())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.output_var, "Summary".to_string(), "Summary");
                            for varname in &self.varnames {
                                ui.selectable_value(&mut self.output_var, varname.clone(), varname.clone());
                            }
                            ui.selectable_value(&mut self.output_var, "Raw".to_string(), "Raw");
                        });
                    if ui.button("Run Monte Carlo").clicked() {
                        self.calc_montecarlo();
                    }
                });

                egui::ScrollArea::vertical()
                .id_salt("output")
                .min_scrolled_height(64.0)
                .show(ui, |ui| {
                    let report =  match self.output_var.as_str() {
                        "Summary" => &self.output_summary,
                        "Raw" => &self.output_raw,
                        s => {
                            let idx: usize = self.varnames.iter().position(|v| v == s).unwrap_or(0);
                            if idx < self.output_reports.len() {
                                &self.output_reports[idx]
                            } else {
                                &self.output_raw
                            }
                        },
                    };
                    ui.add(egui::TextEdit::multiline(&mut report.as_str())
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                    );

                }); //scrollarea
            });  // central

            let (hist, mean, stdev, k): (Option<Histogram>, f64, f64, f64) = match self.output_var.as_str() {
                "Summary" => (None, f64::NAN, f64::NAN, 1.0),
                "Raw" => (None, f64::NAN, f64::NAN, 1.0),
                s => {
                    let idx: usize = self.varnames.iter().position(|v| v == s).unwrap_or(0);
                    if idx < self.output_hists.len() {
                        (self.output_hists[idx].clone(), self.output_gum_means[idx], self.output_gum_stds[idx], self.output_gum_k[idx])
                    } else {
                        (None, f64::NAN, f64::NAN, 1.0)
                    }
                },
            };

            let curve: Option<CurveResult> = match self.output_var.as_str() {
                "Summary" => None,
                "Raw" => None,
                s => {
                    let idx: usize = self.varnames.iter().position(|v| v == s).unwrap_or(0);
                    if idx < self.output_hists.len() {
                        self.output_curves[idx].clone()
                    } else {
                        None
                    }
                }
            };

            egui::SidePanel::right("plot_panel")
            .default_width(200.0)
            .show_inside(ui, |ui| {
                match hist {
                    Some(h) => {
                        let mut bars: Vec<Bar> = Vec::new();
                        for i in 0..h.bins.len() {
                            bars.push(
                                Bar::new(h.bins[i], h.density[i])
                                    .width(h.width)
                                );
                        }
                        
                        Plot::new("my_plot")
                        .x_axis_label("Value")
                        .y_axis_label("Probability Density")
                        .show_grid(false)
                        .show(ui, |plot_ui| {
                            plot_ui.bar_chart(BarChart::new(bars));

                            let c1: Color32 = Color32::from_hex("#BA0C2F").unwrap();
                            let c2: Color32 = Color32::from_hex("#007a86").unwrap();
                            plot_ui.vline(VLine::new(h.low).color(c1).style(LineStyle::Dashed{length: 4.0}));
                            plot_ui.vline(VLine::new(h.high).color(c1).style(LineStyle::Dashed{length: 4.0}));
                            plot_ui.vline(VLine::new(mean-stdev*k).color(c2).style(LineStyle::Dashed{length: 4.0}));
                            plot_ui.vline(VLine::new(mean+stdev*k).color(c2).style(LineStyle::Dashed{length: 4.0}));

                            let pdfx = linspace(mean-stdev*6.0, mean+stdev*6.0, 100);
                            let pdfy: Vec<f64> = pdfx.iter().map(|x| norm_pdf(*x, mean, stdev)).collect();

                            let mut points = Vec::<[f64; 2]>::new();
                            for i in 0..pdfx.len() {
                                points.push((pdfx[i], pdfy[i]).into());
                            }
                            let ppoints = PlotPoints::new(points);
                            let line = Line::new(ppoints).color(c2);
                            plot_ui.line(line);
                        });

                    },
                    None => {},
                };

                match curve {
                    Some(cv) => {
                        Plot::new("curve_plot")
                        .x_axis_label("x")
                        .y_axis_label("y")
                        .show_grid(false)
                        .show(ui, |plot_ui| {
                            let c2: Color32 = Color32::from_hex("#888888").unwrap();

                            let mut xy = Vec::<[f64; 2]>::new();
                            for i in 0..cv.npoints {
                                xy.push((cv.xdata[i], cv.ydata[i]).into());
                            }
                            let line = Points::new(PlotPoints::new(xy)).radius(5.0);
                            plot_ui.points(line);

                            let mut fit = Vec::<[f64; 2]>::new();
                            for i in 0..cv.xplot.dim().1 {
                                fit.push((cv.xplot[i], cv.yplot[i]).into());
                            }
                            let line = Line::new(PlotPoints::new(fit));
                            plot_ui.line(line);

                            let mut c1 = Vec::<[f64; 2]>::new();
                            for i in 0..cv.xplot.dim().1 {
                                c1.push((cv.xplot[i], cv.conf_plus[i]).into());
                            }
                            let line = Line::new(PlotPoints::new(c1)).color(c2).style(LineStyle::Dashed{length:4.0});
                            plot_ui.line(line);

                            let mut c1 = Vec::<[f64; 2]>::new();
                            for i in 0..cv.xplot.dim().1 {
                                c1.push((cv.xplot[i], cv.conf_minus[i]).into());
                            }
                            let line = Line::new(PlotPoints::new(c1)).color(c2).style(LineStyle::Dashed{length:4.0});
                            plot_ui.line(line);

                        });

                    },
                    None => {},
                };


            });  // sidepanel
        });  // window
    }

}

impl eframe::App for SuncalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Main UI update loop
        ctx.set_pixels_per_point(1.5);
        ctx.request_repaint();

        egui::TopBottomPanel::top("top_panel")
            .show(ctx, |ui| {
                self.draw_menu(ui);

        });

        if self.save_window_open {
            self.draw_save_window(ctx);
        }
        if self.load_window_open {
            self.draw_load_window(ctx);
        }

        egui::CentralPanel::default()
            .show(ctx, |ui| {

                if self.model.quantity.len() > 0 {
                    egui::Frame::new()
                    .show(ui, |ui| {
                        self.draw_qty_table(ui);
                    });
                }
                if self.model.curve.len() > 0 {
                    ui.add_space(20.0);
                    ui.separator();
                    egui::Frame::new()
                    .show(ui, |ui| {
                        self.draw_curve_table(ui);
                    });
                }
                if self.model.function.len() > 0 {
                    ui.add_space(20.0);
                    ui.separator();
                    egui::Frame::new()
                    .show(ui, |ui| {
                        self.draw_func_table(ui);
                    });
                };
        });

        if self.show_output {
                self.draw_output(ctx);
        };
    }
}