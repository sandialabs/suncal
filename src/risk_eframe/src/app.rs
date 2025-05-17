// Measurment Decision Risk Application
use std::error::Error;
use eframe::egui;
use eframe::egui::text::{CCursor, CCursorRange};
use egui_plot::{Line, VLine, HLine, Plot, PlotPoints, LineStyle, Legend};
use ecolor::Color32;
use contour::{Contour, ContourBuilder};

use sunlib;
use sunlib::risk::RiskModel;
use sunlib::cfg::{TypeBDist, TypeBNormal, Tolerance, Guardband, GuardbandMethod};
use sunlib::dists::{Distribution, linspace, std_from_itp};


#[derive(PartialEq, Debug)]
enum DistType {
    Normal,
    NormalItp,
    Gamma,
    Uniform,
    Triangular,
}

#[derive(PartialEq, Debug)]
enum TestDistType {
    Normal,
    Gamma,
    Uniform,
    Triangular,
}

#[derive(PartialEq, Debug)]
enum CalcMode {
    Global,
    Conformance,
    Curves
}


#[derive(PartialEq, Debug)]
pub enum RiskCurveOut {
    Pfa,
    Pfr,
    Cpfa,
}


#[derive(PartialEq, Debug)]
pub enum RiskCurveParam {
    Itp,
    Tur,
    Gbf,
    Bias,
}

pub struct SweepParams {
    xvar: RiskCurveParam,
    zvar: RiskCurveParam,
    xstart: f64,
    xstop: f64,
    xnum: usize,
    zvalues: Vec<f64>,

    itp: f64,
    tur: f64,
    gbf: f64,
    gbmethod: GuardbandMethod,
    pbias: f64,
    itp_observed: bool,
}
impl Default for SweepParams {
    fn default() -> Self {
        Self{
            xvar: RiskCurveParam::Itp,
            zvar: RiskCurveParam::Tur,
            xstart: 0.50,
            xstop: 0.95,
            xnum: 20,
            zvalues: vec![1.5, 2.0, 3.0, 4.0],
            itp: 0.95,
            tur: 4.0,
            gbf: 1.0,
            gbmethod: GuardbandMethod::None,
            pbias: 0.0,
            itp_observed: false,
        }
    }
}
impl SweepParams {
    pub fn risk_curve(&self, output: &RiskCurveOut) -> Vec<Vec<[f64; 2]>> {
        let low_lim = -1.0;
        let hi_lim = 1.0;
        let tolerance = Tolerance{low: low_lim, high: hi_lim};

        let mut risk_out: Vec<Vec<[f64; 2]>> = vec![];
        let xvalues = linspace(self.xstart, self.xstop, self.xnum);

        for z in self.zvalues.iter() {
            let mut risk_step: Vec<[f64; 2]> = vec![];

            for x in xvalues.iter() {

                let itp = if self.xvar == RiskCurveParam::Itp {
                    *x
                } else if self.zvar == RiskCurveParam::Itp {
                    *z
                } else {
                    self.itp
                };

                let tur = if self.xvar == RiskCurveParam::Tur {
                    *x
                } else if self.zvar == RiskCurveParam::Tur {
                    *z
                } else {
                    self.tur
                };

                let gbf = if self.xvar == RiskCurveParam::Gbf {
                    *x
                } else if self.zvar == RiskCurveParam::Gbf {
                    *z
                } else {
                    self.gbf
                };

                let pbias = if self.xvar == RiskCurveParam::Bias {
                    *x
                } else if self.zvar == RiskCurveParam::Bias {
                    *z
                } else {
                    self.pbias
                };

                let sigma_meas = 0.5 / tur;
                let mut sigma0 = std_from_itp(itp, 0.0, &tolerance);

                if self.itp_observed {
                    sigma0 = (sigma0.powi(2) - sigma_meas.powi(2)).sqrt();
                }

                let meas_pdf = TypeBDist::Normal(TypeBNormal::new(sigma_meas));
                let process_pdf = Distribution::Normal{mu: pbias, sigma: sigma0};
                let guardband = match self.gbmethod {
                    GuardbandMethod::None => {
                        let acceptance = Tolerance{low: -gbf, high: gbf};
                        Guardband{
                            tolerance: acceptance,
                            method: GuardbandMethod::Manual,
                            target: 0.0,  // NA
                            tur: f64::INFINITY,  // NA
                        }
                    },
                    _ => Guardband{
                        tolerance: tolerance.clone(),
                        method: self.gbmethod.clone(),
                        target: 0.0,   // NA
                        tur: f64::INFINITY, // NA
                    },
                };

                let riskmodel = RiskModel{
                    process: process_pdf,
                    test: meas_pdf,
                    tolerance: tolerance.clone(),
                    guardband: guardband,
                };

                let acceptance = riskmodel.get_guardband();

                let pfx = match output {
                    RiskCurveOut::Pfa => { riskmodel.pfa(&acceptance)*100.0 },
                    RiskCurveOut::Pfr => { riskmodel.pfr(&acceptance)*100.0 },
                    RiskCurveOut::Cpfa => { riskmodel.cpfa(&acceptance)*100.0 },
                };
                risk_step.push( (*x, pfx).into() );
            }
            risk_out.push(risk_step.clone());
        }
        risk_out
    }
}


pub struct RiskApp {
    mode: CalcMode,
    model: RiskModel,
    process: DistType,
    test: TestDistType,
    pvalue1: f64,
    pvalue2: f64,
    tvalue1: f64,
    tvalue2: f64,
    curves: SweepParams,
    zvaluetext: String,
    curveout: RiskCurveOut,
    save_window_open: bool,
    load_window_open: bool,
    config_to_load: String,
    load_message: String,
    result: sunlib::risk::RiskModelResult,
    output: String,
    contours: Vec<Contour>,
    poc_curve: Vec<[f64; 2]>,
    pfa_curve: Vec<Vec<[f64; 2]>>,
}
impl Default for RiskApp {
    fn default() -> Self {
        let mut def = Self {
            mode: CalcMode::Global,
            model: RiskModel::default(),
            process: DistType::Normal,
            test: TestDistType::Normal,
            pvalue1: 0.0,
            pvalue2: 0.51,
            tvalue1: 1.0,  // not used in default Normal
            tvalue2: 0.125,
            curves: SweepParams::default(),
            zvaluetext: String::from("1.5, 2.0, 3.0, 4.0"),
            curveout: RiskCurveOut::Pfa,
            save_window_open: false,
            load_window_open: false,
            config_to_load: String::new(),
            load_message: String::new(),
            result: sunlib::risk::RiskModelResult::default(),
            output: String::from(""),
            contours: vec![],
            poc_curve: vec![],
            pfa_curve: vec![vec![]],
        };
        def.recalc();
        def
    }
}
impl RiskApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }
    fn proc_dist(&self) -> sunlib::dists::Distribution {
        // Get the process Distribution
        match self.process {
            DistType::Normal => sunlib::dists::Distribution::Normal{mu: self.pvalue1, sigma: self.pvalue2},
            DistType::NormalItp => sunlib::dists::Distribution::NormalItp{mu: self.pvalue1, itp: self.pvalue2, low: self.model.tolerance.low, high: self.model.tolerance.high},
            DistType::Uniform => sunlib::dists::Distribution::Uniform{mu: self.pvalue1, a: self.pvalue2},
            DistType::Triangular => sunlib::dists::Distribution::Triangular{mu: self.pvalue1, a: self.pvalue2},
            DistType::Gamma => sunlib::dists::Distribution::Gamma{a: self.pvalue1, b: self.pvalue2},
        }
    }
    fn test_dist(&self) -> sunlib::cfg::TypeBDist {
        // Get the test TypeBDist
        match self.test {
            TestDistType::Normal => sunlib::cfg::TypeBDist::Normal(
                sunlib::cfg::TypeBNormal::new(self.tvalue2)),
            TestDistType::Uniform => sunlib::cfg::TypeBDist::Uniform(
                sunlib::cfg::TypeBUniform{a:self.tvalue2, degf:f64::INFINITY, name:String::new()}),
            TestDistType::Triangular => sunlib::cfg::TypeBDist::Triangular(
                sunlib::cfg::TypeBTriangular{a:self.tvalue2, degf:f64::INFINITY, name:String::new()}),
            TestDistType::Gamma => sunlib::cfg::TypeBDist::Gamma(
                sunlib::cfg::TypeBGamma{a: self.tvalue1, b: self.tvalue2, degf:f64::INFINITY, name:String::new()}),
        }
    }
    fn load_model(&mut self, model: RiskModel) {
        // Load the RiskModel into the UI
        self.model = model;
        match self.model.process {
            sunlib::dists::Distribution::Normal{mu, sigma} => {
                self.process = DistType::Normal;
                self.pvalue1 = mu;
                self.pvalue2 = sigma;
            },
            sunlib::dists::Distribution::NormalItp{itp, mu: _, low: _, high: _} => {
                self.process = DistType::NormalItp;
                self.pvalue1 = itp;
            },
            sunlib::dists::Distribution::Uniform{mu, a} => {
                self.process = DistType::Uniform;
                self.pvalue1 = mu;
                self.pvalue2 = a;
            },
            sunlib::dists::Distribution::Triangular{mu, a} => {
                self.process = DistType::Triangular;
                self.pvalue1 = mu;
                self.pvalue2 = a;
            },
            sunlib::dists::Distribution::Gamma{a, b} => {
                self.process = DistType::Gamma;
                self.pvalue1 = a;
                self.pvalue2 = b;
            },
            _ => { unreachable!(); },
        }
        match &self.model.test {
            sunlib::cfg::TypeBDist::Normal(d) => {
                self.test = TestDistType::Normal;
                self.tvalue2 = d.stddev;
            },
            sunlib::cfg::TypeBDist::Uniform(d) => {
                self.test = TestDistType::Uniform;
                self.tvalue2 = d.a;
            },
            sunlib::cfg::TypeBDist::Triangular(d) => {
                self.test = TestDistType::Triangular;
                self.tvalue2 = d.a;
            },
            sunlib::cfg::TypeBDist::Gamma(d) => {
                self.test = TestDistType::Gamma;
                self.tvalue1 = d.a;
                self.tvalue2 = d.b;
            },
            _ => { unreachable!(); },
        }
    }
    fn recalc(&mut self) {
        // Calculate everything
        self.model.process = self.proc_dist();
        self.model.test = self.test_dist();
        match self.model.check() {
            Err(e) => { self.output = e.to_string() },
            Ok(_) => { 
                self.result = self.model.calculate();
                self.output = self.result.to_string();
            }
        }
        self.contours = match self.build_contours() {
            Ok(c) => c,
            Err(_) => vec![],
        };
    }
    fn build_contours(&self) -> Result<Vec<Contour>, Box<dyn Error>> {
        // Build contour plot curves
        let proc_dist = self.proc_dist();
        let test_dist = self.test_dist();
        let (xlow, xhigh) = proc_dist.domain();
        let tstd = test_dist.std_dev(None);
        let xlow = xlow - tstd*3.0;
        let xhigh = xhigh + tstd*3.0;
        let width = xhigh-xlow;

        let n = 100;  // n x n grid, but in 1D Vec z
        let step = width / n as f64;
        let mut z = Vec::<f64>::new();
        for i in 0..n {
            let x = xlow + i as f64 * step;
            for j in 0..n {
                let y = xlow + j as f64 * step;
                let p = proc_dist.pdf(y) * test_dist.pdf_given_y(y, None).pdf(x);
                z.push(p);
            }
        }

        let c = ContourBuilder::new(n, n, true)
        .x_step(step).y_step(step)
        .x_origin(xlow).y_origin(xlow);

        let zmax = z.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less)).unwrap_or(&0.0);
        let levels = vec![0.001 * zmax, 0.1 * zmax, 0.5 * zmax, 0.9 * zmax];
        Ok(c.contours(&z, &levels)?)
    }

    fn recalc_poc(&mut self) {
        // Calculate probability of conformance
        self.model.test = self.test_dist();
        let tlow = if self.model.tolerance.low.is_finite() {
            self.model.tolerance.low
        } else {
            self.model.tolerance.high - self.model.test.std_dev(None) * 4.0
        };
        let thigh = if self.model.tolerance.high.is_finite() {
            self.model.tolerance.high
        } else {
            self.model.tolerance.low + self.model.test.std_dev(None) * 4.0
        };

        let width = thigh-tlow;
        let tlow = tlow - width/2.0;
        let thigh = thigh + width/2.0;
        let width = thigh-tlow;
        let n = 100;
        let step = width / n as f64;
        let mut poc = Vec::<[f64; 2]>::new();

        for i in 0..n {
            let x = tlow + i as f64 * step;
            poc.push((x, self.model.pr_conform(x)).into());
        }
        self.poc_curve = poc;
    }
    fn calc_curve(&mut self) {
        // Calculate risk Curves
        self.pfa_curve = self.curves.risk_curve(&self.curveout);
    }
    fn calc_curve_zvalues(&mut self) {
        // Split the z-values into float vector
        let svalues = self.zvaluetext.split(&[' ', ',', ';']);
        self.curves.zvalues = svalues.filter_map(|s| s.parse::<f64>().ok()).collect::<Vec<_>>();
        self.calc_curve();
    }
}

impl eframe::App for RiskApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Main UI update loop
        ctx.set_pixels_per_point(1.5);
        ctx.request_repaint();

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
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
                        ui.strong("Measurement Decision Risk Calculator");
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

                if ui.selectable_label(self.mode==CalcMode::Global, "Global Risk").clicked() {
                    self.mode = CalcMode::Global;
                    self.recalc();
                }
                if ui.selectable_label(self.mode==CalcMode::Conformance, "Conformance").clicked() {
                    self.mode = CalcMode::Conformance;
                    self.recalc_poc();
                }
                if ui.selectable_label(self.mode==CalcMode::Curves, "Curves").clicked() {
                    self.mode = CalcMode::Curves;
                    self.calc_curve();
                }
                ui.add_space(62.0);

                if ui.button("📝")
                    .on_hover_text("Export configuration")
                    .clicked() {
                    self.save_window_open = true;
                }
                if ui.button("📓")
                    .on_hover_text("Import configuration")
                    .clicked() {
                        self.load_message = String::new();
                        self.load_window_open = true;
                }
                ui.add_space(62.0);
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        if self.save_window_open {
            self.draw_save_window(ctx);
        }
        if self.load_window_open {
            self.draw_load_window(ctx);
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.save_window_open || self.load_window_open {
                ui.disable();
            }
            if self.mode == CalcMode::Global {
                ui.horizontal_top(|ui| {
                    self.draw_controls(ui);
                    self.draw_output(ui);
                });
                ui.end_row();
                self.draw_plot(ui);
            }
            if self.mode == CalcMode::Conformance {
                self.draw_poc_controls(ui);
                self.draw_poc_plot(ui);
            }
            if self.mode == CalcMode::Curves {
                self.draw_curve_controls(ui);
                self.draw_curve_plot(ui);
            }
        });
    }
}
impl RiskApp {
    fn draw_save_window(&mut self, ctx: &egui::Context ) {
        // Draw the "Save" dialog
        let mut cfg = match self.model.check() {
            Err(e) => e.to_string(),
            Ok(_) => {
                match self.model.get_config() {
                    Ok(m) => m,
                    Err(e) => e.to_string(),
                }
            }
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
        let mut newmodel: Option<RiskModel> = None;

        egui::Window::new("Paste Configuration")
        .open(&mut self.load_window_open)
        .fade_in(true).fade_out(true)
        .show(ctx, |ui| {
            ui.label("Paste a saved configuration here.");
            ui.label(egui::RichText::new(load_msg).color(egui::Color32::from_rgb(255, 0, 0)));
            ui.add(egui::TextEdit::multiline(&mut self.config_to_load)
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .desired_rows(20)
            );
            if ui.button("Import").clicked() {
                let m = RiskModel::load_toml(&self.config_to_load);
                match m {
                    Ok(v) => { newmodel = Some(v); closeme = true; },
                    Err(e) => {self.load_message = e.to_string()},
                }
            }
        });
        if closeme {
            self.load_window_open = false;
            self.load_model(newmodel.unwrap());
            self.recalc();
        }
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        // Global Risk UI
        egui::Grid::new("ctrlgrid").show(ui, |ui| {
            ui.label("Tolerance Low");
            if ui.add(egui::DragValue::new(&mut self.model.tolerance.low).speed(0.01)).changed() {
                self.recalc();
            };
            ui.end_row();
            ui.label("Tolerance High");
            if ui.add(egui::DragValue::new(&mut self.model.tolerance.high).speed(0.01)).changed() {
                self.recalc();
            };
            ui.end_row();

            ui.add(egui::Separator::default());
            ui.add(egui::Separator::default());
            ui.end_row();

            ui.label("Product Distribution");
            egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{:?}", self.process))
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.process, DistType::Normal, "Normal").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.process, DistType::NormalItp, "Normal (ITP)").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.process, DistType::Gamma, "Gamma").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.process, DistType::Uniform, "Uniform").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.process, DistType::Triangular, "Triangular").clicked() { self.recalc() };
                });
            ui.end_row();
            let (label1, label2) = match self.process {
                DistType::Gamma => (String::from("a"), String::from("b")),
                DistType::Normal => (String::from("Mean"), String::from("Std. Dev.")),
                DistType::NormalItp => (String::from("ITP"), String::from("")),
                DistType::Uniform => (String::from("Mean"), String::from("Half-width a")),
                DistType::Triangular => (String::from("Mean"), String::from("Half-width a")),
            };

            ui.label(label1);
            if ui.add(egui::DragValue::new(&mut self.pvalue1).speed(0.01)).changed() {
                    self.recalc();
                };
            ui.end_row();

            match self.process {
                DistType::NormalItp => {},
                _ => {
                    ui.label(label2);
                    if ui.add(egui::DragValue::new(&mut self.pvalue2).speed(0.01)).changed() {
                        self.recalc();
                    };
                    ui.end_row();
                }
            }

            ui.add(egui::Separator::default());
            ui.add(egui::Separator::default());
            ui.end_row();

            ui.label("Test Distribution");
            egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{:?}", self.test))
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.test, TestDistType::Normal, "Normal").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.test, TestDistType::Gamma, "Gamma").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.test, TestDistType::Uniform, "Uniform").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.test, TestDistType::Triangular, "Triangular").clicked() { self.recalc() };
                });
            ui.end_row();

            match self.test {
                TestDistType::Gamma => {
                    ui.label("a");
                    if ui.add(egui::DragValue::new(&mut self.tvalue1).speed(0.01)).changed() {
                        self.recalc();
                        };
                    ui.end_row(); 
                    ui.label("b");
                },
                TestDistType::Uniform => { ui.label("Half-width a"); },
                TestDistType::Triangular => { ui.label("Half-width a"); },
                TestDistType::Normal => { ui.label("Std. Dev."); },
            }

            if ui.add(egui::DragValue::new(&mut self.tvalue2).speed(0.01)).changed() {
                self.recalc();
            };
            ui.end_row();

            ui.add(egui::Separator::default());
            ui.add(egui::Separator::default());
            ui.end_row();

            ui.label("Guardband Method");
            egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{:?}", self.model.guardband.method))
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::None, "None").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::Manual, "Manual").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::Rds, "RDS").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::Dobbert, "Dobbert").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::Pfa, "PFA").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::Cpfa, "Conditional PFA").clicked() { self.recalc() };
                    if ui.selectable_value(&mut self.model.guardband.method, sunlib::cfg::GuardbandMethod::Pfr, "PFR").clicked() { self.recalc() };
            });
            ui.end_row();

            match self.model.guardband.method {
                sunlib::cfg::GuardbandMethod::Manual => {
                    ui.label("Acceptance Low");
                    if ui.add(egui::DragValue::new(&mut self.model.guardband.tolerance.low).speed(0.01)).changed() {
                        self.recalc();
                    };
                    ui.end_row();
                    ui.label("Acceptance High");
                    if ui.add(egui::DragValue::new(&mut self.model.guardband.tolerance.high).speed(0.01)).changed() {
                        self.recalc();
                    };
                    ui.end_row();
                },
                sunlib::cfg::GuardbandMethod::Pfa => {
                    ui.label("Target PFA %");
                    if ui.add(egui::DragValue::new(&mut self.model.guardband.target)
                                .speed(0.001).range(0.0..=1.0)
                                .suffix("%")
                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                .custom_parser(|s| {
                                    let p = s.parse::<f64>();
                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                }))
                                .changed() {
                        self.recalc();
                    }
                }
                sunlib::cfg::GuardbandMethod::Cpfa => {
                    ui.label("Target CPFA %");
                    if ui.add(egui::DragValue::new(&mut self.model.guardband.target)
                                .speed(0.001).range(0.0..=1.0)
                                .suffix("%")
                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                .custom_parser(|s| {
                                    let p = s.parse::<f64>();
                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                }))
                                .changed() {
                        self.recalc();
                    }
                }
                sunlib::cfg::GuardbandMethod::Pfr => {
                    ui.label("Target PFR %");
                    if ui.add(egui::DragValue::new(&mut self.model.guardband.target)
                                .speed(0.001).range(0.0..=1.0)
                                .suffix("%")
                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                .custom_parser(|s| {
                                    let p = s.parse::<f64>();
                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                }))
                                .changed() {
                        self.recalc();
                    }
                }
                _ => {}  // No guardband controls needed
            }
        });
    }
    fn draw_output(&self,  ui: &mut egui::Ui) {
        // Global Risk Text Output
        ui.add(egui::TextEdit::multiline(&mut self.output.clone())
            .font(egui::TextStyle::Monospace)
            .desired_width(f32::INFINITY)
            .desired_rows(11)
        );
    }
    fn draw_plot(&self, ui: &mut egui::Ui) {
        // Global Risk Joint PDF Plot
        let c1: Color32 = Color32::from_hex("#888888").unwrap();
        let c2: Color32 = Color32::from_hex("#007a86").unwrap();
        let c3: Color32 = Color32::from_hex("#cccccc").unwrap();
        let fill1: Color32 = Color32::from_hex("#BA0C2F10").unwrap();
        let fill2: Color32 = Color32::from_hex("#007a8610").unwrap();

        let (rlow, rhigh) = self.result.domain;
        Plot::new("my_plot")
            .x_axis_label("True Value")
            .y_axis_label("Measured Value")
            .show_grid(false)
            .show(ui, |plot_ui| {

            let tol_low = self.model.tolerance.low;
            let tol_high = self.model.tolerance.high;
            let mut top = rhigh;
            let mut bot = rlow;
            let mut left = rlow;
            let mut right = rhigh;
            if tol_high.is_finite() {
                plot_ui.vline(VLine::new(tol_high).color(c1).style(LineStyle::Dashed{length: 4.0}));
                plot_ui.hline(HLine::new(tol_high).color(c1).style(LineStyle::Dashed{length: 4.0}));
                top = tol_high;
                right = tol_high;
            }
            if tol_low.is_finite() {
                plot_ui.vline(VLine::new(tol_low).color(c1).style(LineStyle::Dashed{length: 4.0}));
                plot_ui.hline(HLine::new(tol_low).color(c1).style(LineStyle::Dashed{length: 4.0}));
                bot = tol_low;
                left = tol_low;
            }
            match self.model.guardband.method {
                sunlib::cfg::GuardbandMethod::None => {},
                _ => {
                    if self.result.guardband.low != tol_low {
                        plot_ui.hline(HLine::new(self.result.guardband.low).color(c3).style(LineStyle::Dotted{spacing: 4.0}));
                        bot = self.result.guardband.low;
                    }
                    if self.result.guardband.high != tol_high {
                        plot_ui.hline(HLine::new(self.result.guardband.high).color(c3).style(LineStyle::Dotted{spacing: 4.0}));
                        top = self.result.guardband.high;
                    }
                }
            }
            let poly = egui_plot::Polygon::new(
                PlotPoints::new(
                    vec![[left, bot], [left, top],
                            [rlow, top], [rlow, bot]])
                ).fill_color(fill1);
            plot_ui.add(poly);
            let poly = egui_plot::Polygon::new(
                PlotPoints::new(
                    vec![[right, bot], [right, top],
                            [rhigh, top], [rhigh, bot]])
                ).fill_color(fill1);
            plot_ui.add(poly);
            let poly = egui_plot::Polygon::new(
                PlotPoints::new(
                    vec![[left, top], [right, top],
                            [right, rhigh], [left, rhigh]])
                ).fill_color(fill2);
            plot_ui.add(poly);
            let poly = egui_plot::Polygon::new(
                PlotPoints::new(
                    vec![[left, bot], [right, bot],
                            [right, rlow], [left, rlow]])
                ).fill_color(fill2);
            plot_ui.add(poly);

            for contour in self.contours.iter() {
                let mpoly = contour.geometry();
                for poly in mpoly.iter() {
                    let exterior = poly.exterior();
                    let cont: PlotPoints<'_> = exterior.coords().map(|c| [c.x, c.y]).collect();
                    let line = Line::new(cont).color(c2);
                    plot_ui.line(line);
                }
            }
        });
    }
    fn draw_poc_controls(&mut self, ui: &mut egui::Ui) {
        // Prob. Conformance UI
        egui::Grid::new("pocctrlgrid").show(ui, |ui| {
            ui.label("Tolerance Low");
            if ui.add(egui::DragValue::new(&mut self.model.tolerance.low).speed(0.01)).changed() {
                self.recalc_poc();
            };
            ui.end_row();
            ui.label("Tolerance High");
            if ui.add(egui::DragValue::new(&mut self.model.tolerance.high).speed(0.01)).changed() {
                self.recalc_poc();
            };
            ui.end_row();

            ui.add(egui::Separator::default());
            ui.add(egui::Separator::default());
            ui.end_row();

            ui.label("Test Distribution");
            egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{:?}", self.test))
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.test, TestDistType::Normal, "Normal").clicked() { self.recalc_poc() };
                    if ui.selectable_value(&mut self.test, TestDistType::Gamma, "Gamma").clicked() { self.recalc_poc() };
                    if ui.selectable_value(&mut self.test, TestDistType::Uniform, "Uniform").clicked() { self.recalc_poc() };
                    if ui.selectable_value(&mut self.test, TestDistType::Triangular, "Triangular").clicked() { self.recalc_poc() };
                });
            ui.end_row();

            match self.test {
                TestDistType::Gamma => {
                    ui.label("a");
                    if ui.add(egui::DragValue::new(&mut self.tvalue1).speed(0.01)).changed() {
                        self.recalc_poc();
                        };
                    ui.end_row(); 
                    ui.label("b");
                },
                TestDistType::Uniform => { ui.label("Half-width a"); },
                TestDistType::Triangular => { ui.label("Half-width a"); },
                TestDistType::Normal => { ui.label("Std. Dev."); },
            }

            if ui.add(egui::DragValue::new(&mut self.tvalue2).speed(0.01)).changed() {
                self.recalc_poc();
            };
            ui.end_row();
        });
    }
    fn draw_poc_plot(&self, ui: &mut egui::Ui) {
        // Prob. Conformance plot
        let c1: Color32 = Color32::from_hex("#888888").unwrap();
        Plot::new("poc_plot")
            .x_axis_label("Measured Value")
            .y_axis_label("Probability of Conformance")
            .show_grid(false)
            .show(ui, |plot_ui| {

                let tol_low = self.model.tolerance.low;
                let tol_high = self.model.tolerance.high;
                if tol_high.is_finite() {
                    plot_ui.vline(VLine::new(tol_high).color(c1).style(LineStyle::Dashed{length: 4.0}));
                }
                if tol_low.is_finite() {
                    plot_ui.vline(VLine::new(tol_low).color(c1).style(LineStyle::Dashed{length: 4.0}));
                }
                plot_ui.line(Line::new(PlotPoints::new(self.poc_curve.clone())));
            });
    }
    fn draw_curve_controls(&mut self, ui: &mut egui::Ui) {
        // Prob. Conformance UI
        ui.horizontal(|ui| {
            egui::Grid::new("curvectrlgrid").show(ui, |ui| {
                ui.label("X-Axis");
                egui::ComboBox::new(ui.next_auto_id(), "")
                    .selected_text(RiskApp::curve_label(&self.curves.xvar))
                    .show_ui(ui, |ui| {
                        if ui.selectable_value(&mut self.curves.xvar, RiskCurveParam::Itp, "In-tolerance Probability").clicked() { self.calc_curve() };
                        if ui.selectable_value(&mut self.curves.xvar, RiskCurveParam::Tur, "Test Uncertainty Ratio").clicked() { self.calc_curve() };
                        if ui.selectable_value(&mut self.curves.xvar, RiskCurveParam::Gbf, "Guardband Factor").clicked() { self.calc_curve() };
                        if ui.selectable_value(&mut self.curves.xvar, RiskCurveParam::Bias, "Process Bias").clicked() { self.calc_curve() };
                    });
                ui.end_row();

                ui.label("Z-Axis (Steps)");
                egui::ComboBox::new(ui.next_auto_id(), "")
                    .selected_text(RiskApp::curve_label(&self.curves.zvar))
                    .show_ui(ui, |ui| {
                        if ui.selectable_value(&mut self.curves.zvar, RiskCurveParam::Itp, "In-tolerance Probability").clicked() { self.calc_curve() };
                        if ui.selectable_value(&mut self.curves.zvar, RiskCurveParam::Tur, "Test Uncertainty Ratio").clicked() { self.calc_curve() };
                        if ui.selectable_value(&mut self.curves.zvar, RiskCurveParam::Gbf, "Guardband Factor").clicked() { self.calc_curve() };
                        if ui.selectable_value(&mut self.curves.zvar, RiskCurveParam::Bias, "Process Bias").clicked() { self.calc_curve() };
                    });
                ui.end_row();

                ui.label("X Values");
                ui.horizontal(|ui| {
                    ui.label("Start");
                    if ui.add(egui::DragValue::new(&mut self.curves.xstart).speed(0.01)).changed() {
                        self.calc_curve();
                    };

                    ui.label("Stop");
                    if ui.add(egui::DragValue::new(&mut self.curves.xstop).speed(0.01)).changed() {
                        self.calc_curve();
                    };

                    ui.label("# Points");
                    if ui.add(egui::DragValue::new(&mut self.curves.xnum)).changed() {
                        self.calc_curve();
                    };
                });
                ui.end_row();

                ui.label("Z Values");
                if ui.text_edit_singleline(&mut self.zvaluetext).changed() {
                    self.calc_curve_zvalues();
                }
                ui.end_row();

                ui.label("Output");
                egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{:?}", self.curveout))
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.curveout, RiskCurveOut::Pfa, "PFA").clicked() { self.calc_curve() };
                    if ui.selectable_value(&mut self.curveout, RiskCurveOut::Pfr, "PFR").clicked() { self.calc_curve() };
                    if ui.selectable_value(&mut self.curveout, RiskCurveOut::Cpfa, "CPFA").clicked() { self.calc_curve() };
                });

            });

            ui.separator();

            egui::Grid::new("curvegrid2").show(ui, |ui| {
                if self.curves.xvar != RiskCurveParam::Gbf && self.curves.zvar != RiskCurveParam::Gbf {
                    ui.label("Guardband Method");
                    egui::ComboBox::new(ui.next_auto_id(), "")
                        .selected_text(format!("{:?}", self.curves.gbmethod))
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut self.curves.gbmethod, GuardbandMethod::None, "None").clicked() {self.calc_curve()};
                            if ui.selectable_value(&mut self.curves.gbmethod, GuardbandMethod::Rds, "RDS").clicked() {self.calc_curve()};
                            if ui.selectable_value(&mut self.curves.gbmethod, GuardbandMethod::Rp10, "RP10").clicked() {self.calc_curve()};
                            if ui.selectable_value(&mut self.curves.gbmethod, GuardbandMethod::Dobbert, "Dobbert").clicked() {self.calc_curve()};
                        });
                    ui.end_row();
                }

                if self.curves.xvar != RiskCurveParam::Itp && self.curves.zvar != RiskCurveParam::Itp {
                    ui.label("In Tolerance Probability");
                    if ui.add(egui::DragValue::new(&mut self.curves.itp).speed(0.01)).changed() {
                        self.calc_curve();
                    };
                    ui.end_row();
                }
                if self.curves.xvar != RiskCurveParam::Tur && self.curves.zvar != RiskCurveParam::Tur {
                    ui.label("Test Uncertainty Ratio");
                    if ui.add(egui::DragValue::new(&mut self.curves.tur).speed(0.01)).changed() {
                        self.calc_curve();
                    };
                    ui.end_row();
                }
                if self.curves.xvar != RiskCurveParam::Bias && self.curves.zvar != RiskCurveParam::Bias {
                    ui.label("Process Bias");
                    if ui.add(egui::DragValue::new(&mut self.curves.pbias).speed(0.01)).changed() {
                        self.calc_curve();
                    };
                    ui.end_row();
                }

                if ui.checkbox(&mut self.curves.itp_observed, "Observed ITP").changed() {
                    self.calc_curve();
                };
            });

        });
    }

    fn curve_label(param: &RiskCurveParam) -> String {
        match param {
            RiskCurveParam::Itp => String::from("In Tolerance Probability"),
            RiskCurveParam::Tur => String::from("Test Uncertainty Ratio"),
            RiskCurveParam::Gbf => String::from("Guardband Factor"),
            RiskCurveParam::Bias => String::from("Process Bias %"),
        }
    }
    fn draw_curve_plot(&self, ui: &mut egui::Ui) {
        // Risk Curve plot
        let xlabel = RiskApp::curve_label(&self.curves.xvar);
        let ylabel = match self.curveout {
            RiskCurveOut::Pfa => String::from("Probability of False Accept %"),
            RiskCurveOut::Pfr => String::from("Probability of False Reject %"),
            RiskCurveOut::Cpfa => String::from("Conditional Probability of False Accept %"),
        };

        Plot::new("curve_plot")
            .x_axis_label(xlabel)
            .y_axis_label(ylabel)
            .show_grid(false)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                let mut i = 0;
                for step in &self.pfa_curve {
                    plot_ui.line(
                        Line::new(PlotPoints::new(step.clone()))
                            .name(self.curves.zvalues[i].to_string())
                    );
                    i = i + 1;
                }
            });
    }

}
