// Measurement Quality Assurance Application
use eframe::egui;
use eframe::egui::text::{CCursor, CCursorRange};
use egui_extras::{Column, TableBuilder};
use egui_plot::{Line, VLine, Plot, PlotPoints, LineStyle, Legend};
use ecolor::Color32;


use sunlib;
use sunlib::dists;
use sunlib::cfg::{MeasureSystem, ModelQuantity, Tolerance, Utility, Eopr, TypeBDist, TypeBNormal, TypeBTolerance, Guardband, GuardbandMethod, Calibration, RenewalPolicy, ReliabilityModel, IntervalTarget, Costs};
use sunlib::result::{RiskResult, ReliabilityDecay};

mod examples;


// Things that can't go directly into ModelQuantity
#[derive(PartialEq)]
enum EoprUi {
    True,
    Observed
}

#[derive(PartialEq)]
enum EquipUi {
    Tolerance,
    Symbol,
}

#[derive(PartialEq)]
enum TestIntUi {
    Off,
    Interval,
    Eopr,
}

struct QuantityUi {
    tolerance: ToleranceUi,
    eopr_type: EoprUi,
    eopr_value: f64,
    degrade: ToleranceUi,
    fail: ToleranceUi,
    equip_type: EquipUi,
    equip_tol: f64,
    equip_reliability: f64,
    equip_symbol: String,
    accept_limit: ToleranceUi,
    repair_limit: ToleranceUi,
    prestress_enable: bool,
    poststress_enable: bool,
    prestress_sigma: f64,
    poststress_sigma: f64,
    typebs: Vec<f64>,
    test_interval_type: TestIntUi,
    test_eopr: f64,
    test_interval: f64,  // Input interval to test
    cost_enable: bool,
    costs: Costs,

    tur: f64,
    pfa: f64,
    pfr: f64,
    cpfa: f64,
    bop_dist: Option<dists::Distribution>,
    eop_dist: Option<dists::Distribution>,
    util_dist: Option<dists::Distribution>,
    decay: Option<ReliabilityDecay>,
    calc_interval: f64,  // Calculated interval
}
impl QuantityUi {
    fn update_model(&self, item: &mut ModelQuantity) {
        match &mut item.interval {
            Some(intv) => {
                match self.eopr_type {
                    EoprUi::True => { intv.eopr = Eopr::True(self.eopr_value); },
                    EoprUi::Observed => {intv.eopr = Eopr::Observed(self.eopr_value); },
                };

                match self.test_interval_type {
                    TestIntUi::Off => { intv.target = None; },
                    TestIntUi::Interval => {intv.target = Some(IntervalTarget::Interval(self.test_interval)); },
                    TestIntUi::Eopr => {intv.target = Some(IntervalTarget::Eopr(self.test_eopr)); },
                };

            },
            None => { unreachable!(); },
        };

        match &mut item.utility {
           Some(u) => {
               u.tolerance = self.tolerance.to_tol().unwrap();
               let gbtol = if let Some(gb) = self.accept_limit.to_tol() { gb } else { Tolerance::default() };
               u.guardband.tolerance = gbtol;
               u.degrade = self.degrade.to_tol();
               u.failure = self.fail.to_tol();
           },
           None => {
              item.utility = Some(Utility{
                  tolerance: self.tolerance.to_tol().unwrap(),
                  degrade: self.degrade.to_tol(),
                  failure: self.fail.to_tol(),
                  guardband: Guardband{
                      tolerance:self.accept_limit.to_tol().unwrap(),
                      method: GuardbandMethod::None,
                      target: 0.02,
                      tur: 4.0,
                  },
                  psr: 1.0,
              })
           }
       }

       match self.equip_type {
           EquipUi::Tolerance => {
               item.typeb.clear();
               item.typeb.push(
                   TypeBDist::Tolerance(TypeBTolerance{
                       tolerance: self.equip_tol,
                       kfactor: f64::NAN,
                       confidence: self.equip_reliability,
                       degf: f64::INFINITY,
                       name: String::new(),
                    }));
                },
            EquipUi::Symbol => {
                item.typeb.clear();
                item.typeb.push(
                    TypeBDist::Symbol(self.equip_symbol.clone())
                );
            },
        };

        for bsigma in &self.typebs {
            item.typeb.push(TypeBDist::Normal(TypeBNormal{stddev: *bsigma, degf: f64::INFINITY, name: String::new()}));
        };

        let prestress = if self.prestress_enable {
            Some(TypeBDist::Normal(
                TypeBNormal{
                    stddev: self.prestress_sigma,
                    degf: f64::INFINITY,
                    name: String::new(),
                }))
        } else {
            None
        };
        let poststress = if self.poststress_enable {
            Some(TypeBDist::Normal(
                TypeBNormal{
                    stddev: self.poststress_sigma,
                    degf: f64::INFINITY,
                    name: String::new(),
                }))
        } else {
            None
        };

       match &mut item.calibration {
           Some(calib) => {
               calib.prestress = prestress;
               calib.poststress = poststress;
               if self.repair_limit.enable {
                    calib.repair = self.repair_limit.to_tol();
               } else {
                   calib.repair = None;
               };
            },
            None => {

                item.calibration = Some(Calibration{
                    policy: RenewalPolicy::Never,
                    repair: None,
                    prob_discard: 0.0,
                    prestress: prestress,
                    poststress: poststress,
                    mte_adjust: None,
                    mte_repair: None,
                    reliability_model: ReliabilityModel::None,
                });
            },
       };

       if self.cost_enable {
           item.cost = Some(self.costs.clone());
       } else {
           item.cost = None;
       };
    }
}


pub struct ColumnEnable {
    utility: bool,
    measure: bool,
    cost: bool,
    tur: bool,
    pfa: bool,
    pfr: bool,
    cpfa: bool,
}
impl ColumnEnable {
    fn new() -> Self {
        Self{
            utility: false,
            measure: false,
            cost: false,
            tur: true,
            pfa: true,
            pfr: false,
            cpfa: false,
        }
    }
}

#[derive(PartialEq, Debug)]
enum PlotType {
    Reliability,
    Utility,
    Decay,
}

struct PlotData {
    plot_type: PlotType,
    qty: usize,
    data1: Vec<[f64; 2]>,
    data2: Vec<[f64; 2]>,
    vline1: f64,
    vline2: f64,
}

pub struct MqaApp {
    model: MeasureSystem,
    quantities: Vec<QuantityUi>,
    output: String,
    save_window_open: bool,
    load_window_open: bool,
    show_output: bool,
    show_plots: bool,
    plot: PlotData,
    config_to_load: String,
    load_message: String,
    columns: ColumnEnable,
}
impl Default for MqaApp {
    fn default() -> Self {
        let mut def = Self {
            model: MeasureSystem::default(),
            quantities: vec![],
            output: String::new(),
            save_window_open: false,
            load_window_open: false,
            show_output: false,
            show_plots: false,
            plot: PlotData{
                plot_type: PlotType::Reliability,
                qty: 0,
                data1: vec![],
                data2: vec![],
                vline1: -1.0,
                vline2: 1.0,
            },
            config_to_load: String::new(),
            load_message: String ::new(),
            columns: ColumnEnable::new(),
        };
        def.recalc();  // Recalc all or 0 - or not needed
        def
    }
}
impl MqaApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }
    fn recalc(&mut self) {

        if self.quantities.len() > 0 {
            for i in 0..self.quantities.len() {
                self.quantities[i].update_model(&mut self.model.quantity[i]);
            }
        }

        // Calculate everything
        let result = self.model.calculate();
        self.output = match result {
            Ok(r) => {

                if self.quantities.len() > 0 {
                    for i in 0..self.quantities.len() {
                        self.quantities[i].calc_interval = r.quantities[i].interval;
                        match &r.quantities[i].risk {
                            Some(risk) => {
                                match risk {
                                    RiskResult::Global(rr) => {
                                        self.quantities[i].pfa = rr.pfa_true;
                                        self.quantities[i].pfr = rr.pfr_true;
                                        self.quantities[i].cpfa = rr.cpfa_true;
                                        self.quantities[i].tur = rr.tur;
                                        self.quantities[i].accept_limit = ToleranceUi::from_tol(&rr.acceptance.clone());
                                    },
                                    _ => { unreachable!() },
                                }
                            },
                            _ => { unreachable!() },
                        }
                        match &r.quantities[i].reliability {
                            Some(rel) => {
                                self.quantities[i].bop_dist = Some(rel.pdf_bop.clone());
                                self.quantities[i].eop_dist = Some(rel.pdf_eop.clone());
                                self.quantities[i].util_dist = Some(rel.utility.clone());
                                self.quantities[i].decay = rel.decay.clone();
                            }
                            _ => {
                                self.quantities[i].bop_dist = None;
                                self.quantities[i].eop_dist = None;
                                self.quantities[i].util_dist = None;
                                self.quantities[i].decay = None;
                            },
                        }

                    }
                }

                r.to_string()
            },
            Err(e) => {
                e.to_string()
            }
        };
        if self.show_plots {
            self.replot();
        }
    }
    fn load_model(&mut self, model: MeasureSystem) {
        self.model = model;
        self.quantities.clear();

        for qty in &self.model.quantity {

            if let Some(_) = &qty.utility {
                self.columns.utility = true;
            };
            if let Some(_) = &qty.cost {
                self.columns.cost = true;
            };
            if let Some(_) = &qty.calibration {
                self.columns.measure = true;
            };


            let (eoprtype, eoprval) = match &qty.interval.as_ref().unwrap().eopr {
                Eopr::True(x) => (EoprUi::True, x),
                Eopr::Observed(x) => (EoprUi::Observed, x),
            };


            let (equip_type, equip_tol, equip_rel, equip_sym) = if qty.typeb.len() > 0 {
                match &qty.typeb[0] {
                    TypeBDist::Tolerance(btol) => {
                        (EquipUi::Tolerance,
                         btol.tolerance,
                         btol.confidence,
                         String::new()
                        )
                    },
                    TypeBDist::Symbol(sym) => {
                        (EquipUi::Symbol, 0.25, 0.95, sym.to_string())
                    },
                    _ => todo!(),
                }
            } else {
                (EquipUi::Tolerance, 0.25, 0.9545, String::new())
            };

            let mut typebs = vec![];
            for (i, typeb) in qty.typeb.iter().enumerate() {
                if i > 0 {
                    match typeb {
                        TypeBDist::Normal(v) => { typebs.push(v.stddev); }
                        _ => {},
                    };
                };
            }

            let utility = &qty.utility.as_ref().unwrap();
            //let degrade = match &qty.utility.as_ref().unwrap().degrade {
            let degrade = match &utility.degrade {
                Some(d) => ToleranceUi::from_tol(d),
                None => ToleranceUi::new(false),
            };
            //let fail = match &qty.utility.as_ref().unwrap().failure {
            let fail = match &utility.failure {
                Some(d) => ToleranceUi::from_tol(d),
                None => ToleranceUi::new(false),
            };
            let accept_limit = ToleranceUi::from_tol(&utility.guardband.tolerance);

            let repair = match &qty.calibration {
                Some(calib) => {
                    match &calib.repair {
                        Some(rep) => ToleranceUi::from_tol(&rep),
                        None => ToleranceUi::new(false),
                    }
                },
                None => ToleranceUi::new(false),
            };

            let (costs, cost_enable) = match &qty.cost {
                Some(c) => (c.clone(), true),
                None => (Costs::default(), false),
            };

            let (testint_type, testint, testeopr) = match &qty.interval.as_ref().unwrap().target {
                Some(target) => {
                    match target {
                        IntervalTarget::Interval(f) => (TestIntUi::Interval, *f, 0.9),
                        IntervalTarget::Eopr(f) => (TestIntUi::Eopr, 1.0, *f),
                    }
                },
                None => (TestIntUi::Off, 1.0, 0.9),
            };

            let (prestress_enable, prestress) = match &qty.calibration {
                Some(calib) => {
                    match &calib.prestress {
                        Some(p) => {
                            match p {
                                TypeBDist::Normal(v) => (true, v.stddev),
                                _ => unreachable!(),
                            }
                        },
                        None => (false, 1.0),
                    }
                },
                None => (false, 1.0),
            };

            let (poststress_enable, poststress) = match &qty.calibration {
                Some(calib) => {
                    match &calib.poststress {
                        Some(p) => {
                            match p {
                                TypeBDist::Normal(v) => (true, v.stddev),
                                _ => todo!(),
                            }
                        },
                        None => (false, 1.0),
                    }
                },
                None => (false, 1.0),
            };

            self.quantities.push(
                QuantityUi{
                    tolerance: ToleranceUi::from_tol(&qty.utility.as_ref().unwrap().tolerance),
                    eopr_type: eoprtype,
                    eopr_value: *eoprval,
                    degrade: degrade,
                    fail: fail,
                    equip_type: equip_type,
                    equip_tol: equip_tol,
                    equip_reliability: equip_rel,
                    equip_symbol: equip_sym,
                    accept_limit: accept_limit,
                    repair_limit: repair,
                    prestress_enable: prestress_enable,
                    prestress_sigma: prestress,
                    poststress_enable: poststress_enable,
                    poststress_sigma: poststress,
                    typebs: typebs,
                    test_interval_type: testint_type,
                    test_eopr: testeopr,
                    test_interval: testint,
                    cost_enable: cost_enable,
                    costs: costs,
                    pfa: 0.0,
                    pfr: 0.0,
                    cpfa: 0.0,
                    tur: 0.0,
                    bop_dist: None,
                    eop_dist: None,
                    util_dist: None,
                    decay: None,
                    calc_interval: 0.0,
                }
            );
        }
    }
}

impl eframe::App for MqaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Main UI update loop
        ctx.set_pixels_per_point(1.5);
        ctx.request_repaint();

        egui::TopBottomPanel::top("top_panel")
            .show(ctx, |ui| {
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
                        ui.strong("End-to-end Measurement Quality Assurance");
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
                ui.add_space(20.0);

                if ui.button("⊞").on_hover_text("Add Quantity").clicked() {
                    let newq = ModelQuantity::new_mqa();
                    self.quantities.push(
                        QuantityUi{
                            tolerance: ToleranceUi::from_tol(&newq.utility.as_ref().unwrap().tolerance),
                                         eopr_type: EoprUi::Observed,
                                         eopr_value: 0.95,
                                         degrade: ToleranceUi::new(false),
                                         fail: ToleranceUi::new(false),
                                         equip_type: EquipUi::Tolerance,
                                         equip_tol: 0.25,
                                         equip_reliability: 0.9545,
                                         equip_symbol: String::new(),
                                         accept_limit: ToleranceUi::new(false),
                                         repair_limit: ToleranceUi::new(false),
                                         prestress_enable: false,
                                         poststress_enable: false,
                                         prestress_sigma: 1.0,
                                         poststress_sigma: 1.0,
                                         typebs: vec![],
                                         test_interval_type: TestIntUi::Off,
                                         test_eopr: 0.95,
                                         test_interval: 1.0,
                                         cost_enable: false,
                                         costs: Costs::default(),
                                         tur: 0.0,
                                         pfa: 0.0,
                                         pfr: 0.0,
                                         cpfa: 0.0,
                                         bop_dist: None,
                                         eop_dist: None,
                                         util_dist: None,
                                         decay: None,
                                         calc_interval: 0.0,
                        }
                    );
                    self.model.quantity.push(
                        newq
                    );
                };
                if ui.button("⊟").on_hover_text("Remove quantity").clicked() {
                    self.model.quantity.pop();
                    self.quantities.pop();
                };

                ui.add_space(20.0);

                let response = ui.button("⛭").on_hover_text("Show/Hide Columns");
                let popup_id = ui.make_persistent_id("column_setup");
                if response.clicked() {
                    ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                }
                egui::popup::popup_above_or_below_widget(
                    ui, popup_id, &response,
                    egui::AboveOrBelow::Below,
                    egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                        ui.set_min_width(180.0);
                        ui.label("Show:");
                        ui.checkbox(&mut self.columns.utility, "End-Item Utility");
                        ui.checkbox(&mut self.columns.measure, "Measurement & Interval");
                        ui.checkbox(&mut self.columns.cost, "Cost Model");
                        ui.checkbox(&mut self.columns.tur, "Test Uncertainty Ratio");
                        ui.checkbox(&mut self.columns.pfa, "Probability of False Accept");
                        ui.checkbox(&mut self.columns.cpfa, "Conditional Probability of False Accept");
                        ui.checkbox(&mut self.columns.pfr, "Probability of False Reject");
                    });

                if ui.toggle_value(&mut self.show_output, "🖹").on_hover_text("Show raw output").changed() {
                    if self.show_output { self.show_plots = false; }
                };
                if ui.toggle_value(&mut self.show_plots, "📈").on_hover_text("Show plots").changed() {
                    if self.show_plots { self.show_output = false; self.replot(); }
                };
                ui.add_space(62.0);

                egui::widgets::global_theme_preference_buttons(ui);
                ui.add_space(62.0);
            });

            if self.save_window_open || self.load_window_open {
                ui.disable();
            }
        });

        if self.save_window_open {
            self.draw_save_window(ctx);
        }
        if self.load_window_open {
            self.draw_load_window(ctx);
        }

        egui::CentralPanel::default()
            .show(ctx, |ui| {
            self.draw_table(ui);
        });

        egui::TopBottomPanel::bottom("bottom_panel")
            .min_height(32.0)
            .resizable(true)
            .show(ctx, |ui| {
            self.draw_output(ui);
        });

    
    }
}
impl MqaApp {
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
            ui.horizontal(|ui| {
                if ui.button("RP19 Solar Example").clicked() {
                    self.config_to_load = examples::SOLAR.to_string();
                }
                if ui.button("RP19 Cannon Example").clicked() {
                    self.config_to_load = examples::CANNON.to_string();
                }
                if ui.button("RP19 Altimeter Example").clicked() {
                    self.config_to_load = examples::ALTIMETER.to_string();
                }
            });
        });
        if closeme {
            self.load_window_open = false;
            self.load_model(newmodel.unwrap());
            self.recalc();  // ALL
        }
    }

    fn draw_table(&mut self, ui: &mut egui::Ui) {
        let mut need_recalc = false;

        let mut colcnt = 6;
        if self.columns.utility { colcnt += 1; }
        if self.columns.measure { colcnt += 1; }
        if self.columns.cost { colcnt += 1; }
        if self.columns.tur { colcnt += 1; }
        if self.columns.pfa { colcnt += 1; }
        if self.columns.cpfa { colcnt += 1; }
        if self.columns.pfr { colcnt += 1; }

        let table = TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .columns(Column::initial(90.0), colcnt);

        table
            .header(20.0, |mut header| {
                header.col(|ui| {
                        ui.strong("Symbol");
                });
                header.col(|ui| {
                    ui.strong("Testpoint").on_hover_ui(|ui| {
                        ui.label("Expected or nominal value");
                    });
                });
                header.col(|ui| {
                    ui.strong("Tolerance");
                });
                if self.columns.utility {
                    header.col(|ui| {
                        ui.strong("Utility");
                    });
                };
                header.col(|ui| {
                    ui.strong("EOPR");
                });
                header.col(|ui| {
                    ui.strong("Equipment");
                });
                header.col(|ui| {
                    ui.strong("Guardband");
                });
                if self.columns.measure {
                    header.col(|ui| {
                        ui.strong("Measurement");
                    });
                };
                if self.columns.cost {
                    header.col(|ui| {
                        ui.strong("Costs");
                    });
                };
                if self.columns.tur {
                    header.col(|ui| {
                        ui.strong("TUR");
                    });
                };
                if self.columns.pfa {
                    header.col(|ui| {
                        ui.strong("PFA");
                    });
                };
                if self.columns.cpfa {
                    header.col(|ui| {
                        ui.strong("CPFA");
                    });
                };
                if self.columns.pfr {
                    header.col(|ui| {
                        ui.strong("PFR");
                    });
                };

            })
            .body(|body| {
                body.rows(20.0, self.model.quantity.len(), |mut row| {
                    let idx = row.index();
                    let item = &mut self.model.quantity[idx];

                    // SYMBOL
                    row.col(|ui| {
                        ui.text_edit_singleline(&mut item.symbol);
                    });

                    // TESTPOINT
                    row.col(|ui| {
                        if ui.add(egui::DragValue::new(&mut item.measured).speed(0.01)).changed() {
                            need_recalc = true;
                        }
                    });

                    // TOLERANCE
                    row.col(|ui| {
                        if tolerance_widget(ui, &mut self.quantities[idx].tolerance) {
                            need_recalc = true;
                        }
                    });

                    // UTILITY
                    if self.columns.utility {
                        row.col(|ui| {
                            let label = if self.quantities[idx].degrade.enable {
                                self.quantities[idx].degrade.to_string()
                            } else {
                                String::new()
                            };
                            ui.horizontal(|ui| {
                                ui.label(label);
                                let response = ui.button("⏷");
                                let popup_id = ui.make_persistent_id("util_pop");
                                if response.clicked() {
                                    ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                                }
                                egui::popup::popup_above_or_below_widget(
                                    ui, popup_id, &response,
                                    egui::AboveOrBelow::Below,
                                    egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                                        ui.set_min_width(100.0);
                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut self.quantities[idx].degrade.enable, "Degrade Limit").changed() {
                                                need_recalc = true;
                                            };
                                            if tolerance_widget(ui, &mut self.quantities[idx].degrade) {
                                                need_recalc = true;
                                            }
                                        });
                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut self.quantities[idx].fail.enable, "Failure Limit").changed() {
                                                need_recalc = true;
                                            };
                                            if tolerance_widget(ui, &mut self.quantities[idx].fail) {
                                                need_recalc = true;
                                            }
                                        });
                                        ui.horizontal(|ui| {
                                            if let Some(ref mut utility) = item.utility {
                                                ui.label("Successful Outcome Probability");
                                                if ui.add(egui::DragValue::new(&mut utility.psr)
                                                    .speed(0.001).range(0.0..=1.0)
                                                    .suffix("%")
                                                    .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                    .custom_parser(|s| {
                                                        let p = s.parse::<f64>();
                                                        match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                    })
                                                ).changed() {
                                                    need_recalc = true;
                                                };
                                            };
                                        })
                                    });
                            });
                        });
                    };

                    // EOPR
                    row.col(|ui| {
                        match &mut item.interval {
                            Some(_) => {
                                ui.horizontal(|ui| {
                                    if ui.add(egui::DragValue::new(&mut self.quantities[idx].eopr_value)
                                        .speed(0.001).range(0.0..=1.0)
                                        .suffix("%")
                                        .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                        .custom_parser(|s| {
                                            let p = s.parse::<f64>();
                                            match p { Ok(v) => Some(v / 100.0), _ => None,}
                                        })
                                    ).changed() {
                                        need_recalc = true;
                                    };


                                    let response = ui.button("⏷");
                                    let popup_id = ui.make_persistent_id("eopr_pop");
                                    if response.clicked() {
                                        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                                    }
                                    egui::popup::popup_above_or_below_widget(
                                        ui, popup_id, &response,
                                        egui::AboveOrBelow::Below,
                                        egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                                        ui.set_min_width(100.0);
                                        if ui.radio_value(&mut self.quantities[idx].eopr_type, EoprUi::True, "True").changed() {
                                            need_recalc = true;
                                        };
                                        if ui.radio_value(&mut self.quantities[idx].eopr_type, EoprUi::Observed, "Observed").changed() {
                                            need_recalc = true;
                                        };
                                    });

                                });

                            },
                            None => {ui.label("No EOPR");},
                        }
                    });


                    // EQUIPMENT
                    row.col(|ui| {
                        ui.horizontal(|ui| {
                            if self.quantities[idx].equip_type == EquipUi::Symbol {
                                if ui.add(egui::TextEdit::singleline(&mut self.quantities[idx].equip_symbol).desired_width(50.0)).changed() {
                                    need_recalc = true;
                                };
                            } else {
                                ui.label("±");
                                if ui.add(egui::DragValue::new(&mut self.quantities[idx].equip_tol).speed(0.01)).changed() {
                                        need_recalc = true;
                                    };
                                ui.label("@");
                                if ui.add(egui::DragValue::new(&mut self.quantities[idx].equip_reliability)
                                    .speed(0.001).range(0.0..=1.0)
                                    .suffix("%")
                                    .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                    .custom_parser(|s| {
                                        let p = s.parse::<f64>();
                                        match p { Ok(v) => Some(v / 100.0), _ => None,}
                                    }))
                                    .changed() {
                                        need_recalc = true;
                                    };
                                    ui.label("%");
                            };

                            let response = ui.button("⏷");
                            let popup_id = ui.make_persistent_id("equip_pop");
                            if response.clicked() {
                                ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                            }
                            egui::popup::popup_above_or_below_widget(
                                ui, popup_id, &response,
                                egui::AboveOrBelow::Below,
                                egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                                    ui.set_min_width(125.0);
                                    if ui.radio_value(&mut self.quantities[idx].equip_type, EquipUi::Tolerance, "Tolerance").changed() {
                                        need_recalc = true;
                                    };
                                    if ui.radio_value(&mut self.quantities[idx].equip_type, EquipUi::Symbol, "Another Quantity").changed() {
                                        need_recalc = true;
                                    };
                                });
                            });
                    });

                    // GUARDBAND
                    row.col(|ui| {
                        if let Some(ref mut utility) = item.utility {
                            let gblabel = match &utility.guardband.method {
                                    GuardbandMethod::Manual => { self.quantities[idx].accept_limit.to_string() },
                                    GuardbandMethod::Rds => { self.quantities[idx].accept_limit.to_string() },
                                    GuardbandMethod::Pfa => { self.quantities[idx].accept_limit.to_string() },
                                    GuardbandMethod::Cpfa => { self.quantities[idx].accept_limit.to_string() },
                            _ => { String::from("None") },
                            };
                            ui.horizontal(|ui| {

                                if utility.guardband.method == GuardbandMethod::Manual {
                                    if tolerance_widget(ui, &mut self.quantities[idx].accept_limit ) {
                                        need_recalc = true;
                                    }
                                } else {
                                    ui.label(gblabel);
                                };

                                let response = ui.button("⏷");
                                let popup_id = ui.make_persistent_id("gb_pop");
                                if response.clicked() {
                                    ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                                }
                                egui::popup::popup_above_or_below_widget(
                                    ui, popup_id, &response,
                                    egui::AboveOrBelow::Below,
                                    egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                                        ui.set_min_width(125.0);
                                        if ui.radio_value(&mut utility.guardband.method, GuardbandMethod::None, "None").changed() {
                                            need_recalc = true;
                                        };
                                        if ui.radio_value(&mut utility.guardband.method, GuardbandMethod::Manual, "Manual").changed() {
                                            need_recalc = true;
                                        };
                                        ui.horizontal(|ui| {
                                            if ui.radio_value(&mut utility.guardband.method, GuardbandMethod::Rds, "RDS").changed() {
                                                need_recalc = true;
                                            };
                                            ui.label("Minimum TUR:");
                                            if ui.add(egui::DragValue::new(&mut utility.guardband.tur).speed(0.01)).changed() { need_recalc = true;}
                                        });
                                        ui.horizontal(|ui| {
                                            if ui.radio_value(&mut utility.guardband.method, GuardbandMethod::Pfa, "PFA").changed() {
                                                need_recalc = true;
                                            };
                                            ui.label("Target:");
                                            if ui.add(egui::DragValue::new(&mut utility.guardband.target)                                                    .speed(0.001).range(0.0..=1.0)
                                                .suffix("%")
                                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                .custom_parser(|s| {
                                                    let p = s.parse::<f64>();
                                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                })).changed() { need_recalc = true;}
                                        });
                                        ui.horizontal(|ui| {
                                            if ui.radio_value(&mut utility.guardband.method, GuardbandMethod::Cpfa, "CPFA").changed() {
                                                need_recalc = true;
                                            };
                                            ui.label("Target:");
                                            if ui.add(egui::DragValue::new(&mut utility.guardband.target)                                                    .speed(0.001).range(0.0..=1.0)
                                                .suffix("%")
                                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                .custom_parser(|s| {
                                                    let p = s.parse::<f64>();
                                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                })).changed() { need_recalc = true;}
                                        });
                                    });
                                });
                            };
                    });

                    // MEASUREMENT
                    if self.columns.measure {
                        row.col(|ui| {
                            if let Some(ref mut interval) = item.interval {
                                if let Some(ref mut calib) = item.calibration {

                                ui.horizontal(|ui| {
                                    ui.label(format!("{} yr", interval.years));

                                    let response = ui.button("⏷");
                                    let popup_id = ui.make_persistent_id("meas_pop");
                                    if response.clicked() {
                                        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                                    }
                                    egui::popup::popup_above_or_below_widget(
                                        ui, popup_id, &response,
                                        egui::AboveOrBelow::Below,
                                        egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                                            ui.set_min_width(125.0);

                                            egui::Grid::new("some_unique_id").show(ui, |ui| {
                                                ui.label("Renewal Policy");
                                                if ui.radio_value(&mut calib.policy, RenewalPolicy::Never, "Never").changed() { need_recalc = true; };
                                                ui.end_row(); ui.label("");
                                                if ui.radio_value(&mut calib.policy, RenewalPolicy::Always, "Always").changed() { need_recalc = true; };
                                                ui.end_row(); ui.label("");
                                                if ui.radio_value(&mut calib.policy, RenewalPolicy::Asneeded, "As-Needed").changed() { need_recalc = true; };
                                                ui.end_row();

                                                ui.label("Repair Limit");
                                                if ui.checkbox(&mut self.quantities[idx].repair_limit.enable, "").changed() { need_recalc = true; };
                                                if tolerance_widget(ui, &mut self.quantities[idx].repair_limit) { need_recalc = true; };
                                                ui.end_row();

                                                ui.label("Probability of discarding OOT");
                                                if ui.add(egui::DragValue::new(&mut calib.prob_discard)
                                                    .speed(0.001).range(0.0..=1.0)
                                                    .suffix("%")
                                                    .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                    .custom_parser(|s| {
                                                        let p = s.parse::<f64>();
                                                        match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                    })
                                                ).changed() {
                                                    need_recalc = true;
                                                };
                                                ui.end_row();

                                                ui.label("Pre-test Stress Std. Dev.");
                                                ui.horizontal(|ui| {
                                                   if ui.checkbox(&mut self.quantities[idx].prestress_enable, "").changed() { need_recalc = true; }
                                                   if ui.add(egui::DragValue::new(&mut self.quantities[idx].prestress_sigma).speed(0.01)).changed() { need_recalc = true; }
                                                });
                                                ui.end_row();
                                                ui.label("Post-test Stress Std. Dev.");
                                                ui.horizontal(|ui| {
                                                    if ui.checkbox(&mut self.quantities[idx].poststress_enable, "").changed() { need_recalc = true; }
                                                    if ui.add(egui::DragValue::new(&mut self.quantities[idx].poststress_sigma).speed(0.01)).changed() { need_recalc = true; }
                                                });
                                                ui.end_row();
                                                ui.label("Additional Uncertainties (Std. Dev)");
                                                ui.horizontal(|ui| {
                                                    if ui.button("+").clicked() {
                                                        self.quantities[idx].typebs.push(1.0);
                                                        need_recalc = true;
                                                    };
                                                    if ui.button("-").clicked() {
                                                        self.quantities[idx].typebs.pop();
                                                        need_recalc = true;
                                                    };
                                                });
                                                ui.end_row();
                                                for i in 0..self.quantities[idx].typebs.len() {
                                                    ui.label("");
                                                    if ui.add(egui::DragValue::new(&mut self.quantities[idx].typebs[i]).speed(0.01)).changed() { need_recalc = true; }
                                                    ui.end_row();
                                                }

                                                ui.separator();
                                                ui.separator();
                                                ui.end_row();

                                                ui.label("Reliability Model");
                                                if ui.radio_value(&mut calib.reliability_model, ReliabilityModel::None, "None").changed() { need_recalc = true; }
                                                ui.end_row(); ui.label("");
                                                if ui.radio_value(&mut calib.reliability_model, ReliabilityModel::Exponential, "Exponential").changed() { need_recalc = true; }
                                                ui.end_row(); ui.label("");
                                                if ui.radio_value(&mut calib.reliability_model, ReliabilityModel::RandomWalk, "Random Walk").changed() { need_recalc = true; }
                                                ui.end_row();

                                                ui.label("Observed Interval (years)");
                                                if ui.add(egui::DragValue::new(&mut interval.years).speed(0.1)).changed() { need_recalc = true; };
                                                ui.end_row();
                                                ui.separator();
                                                ui.separator();
                                                ui.end_row();

                                                ui.strong("Test Interval:");
                                                ui.end_row();
                                                if ui.radio_value(&mut self.quantities[idx].test_interval_type, TestIntUi::Off, "None").changed() { need_recalc = true; }
                                                ui.end_row();
                                                if ui.radio_value(&mut self.quantities[idx].test_interval_type, TestIntUi::Interval, "Interval Target").changed() { need_recalc = true; }
                                                if ui.add_enabled(
                                                    self.quantities[idx].test_interval_type == TestIntUi::Interval,
                                                    egui::DragValue::new(&mut self.quantities[idx].test_interval).speed(0.1)).changed() { need_recalc = true; }
                                                ui.end_row();
                                                if ui.radio_value(&mut self.quantities[idx].test_interval_type, TestIntUi::Eopr, "EOPR Target").changed() { need_recalc = true; }
                                                if ui.add_enabled(
                                                    self.quantities[idx].test_interval_type == TestIntUi::Eopr,
                                                    egui::DragValue::new(&mut self.quantities[idx].test_eopr)
                                                    .speed(0.001).range(0.0..=1.0)
                                                    .suffix("%")
                                                    .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                    .custom_parser(|s| {
                                                        let p = s.parse::<f64>();
                                                        match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                    })
                                                ).changed() { need_recalc = true; }
                                                ui.end_row();


                                            });

                                        });
                                    });
                                };
                            };
                        });
                    };

                    // COST
                    if self.columns.cost {
                        row.col(|ui| {

                            ui.horizontal(|ui| {
                            ui.label(if self.quantities[idx].cost_enable {"Enabled"} else {""});

                            let response = ui.button("⏷");
                            let popup_id = ui.make_persistent_id("cost_pop");
                            if response.clicked() {
                                ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                            }
                            egui::popup::popup_above_or_below_widget(
                                ui, popup_id, &response,
                                egui::AboveOrBelow::Below,
                                egui::popup::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                                    ui.set_min_width(200.0);

                                    if ui.checkbox(&mut self.quantities[idx].cost_enable, "Enable Cost Model").changed() { need_recalc = true; };
                                    ui.separator();

                                    if self.quantities[idx].cost_enable {
                                        egui::Grid::new("some_unique_id").show(ui, |ui| {

                                            ui.strong("Calibration Costs");
                                            ui.end_row();

                                            ui.label("Calibration");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.cal)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.label("Adjustment");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.adjust)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.label("Repair");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.repair)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.label("UUTs in Inventory");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.num_uuts)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.separator();
                                            ui.end_row();
                                            ui.strong("Spare UUTs");
                                            ui.end_row();

                                            ui.label("Spares Coverage");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.spare_factor).speed(0.01)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.label("Cost of a UUT");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.new_uut)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.label("Spare Startup Cost");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.spare_startup)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.separator();
                                            ui.end_row();
                                            ui.strong("Downtimes (days)");
                                            ui.end_row();

                                            ui.label("Calibration");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.down_cal)).changed() { need_recalc = true; }
                                            ui.end_row();
                                            ui.label("Adjustment");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.down_adj)).changed() { need_recalc = true; }
                                            ui.end_row();
                                            ui.label("Repair");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.down_rep)).changed() { need_recalc = true; }
                                            ui.end_row();

                                            ui.separator();
                                            ui.end_row();
                                            ui.strong("End-Item Performance");
                                            ui.end_row();
                                            ui.label("Cost of unsuccessful outcome");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.cost_fa)).changed() { need_recalc = true; }
                                            ui.end_row();
                                            ui.label("Probability of unsuccessful outcome given failure");
                                            if ui.add(egui::DragValue::new(&mut self.quantities[idx].costs.p_use)
                                                .speed(0.001).range(0.0..=1.0)
                                                .suffix("%")
                                                .custom_formatter(|n, _| {let p = n * 100.0; format!("{p:.2}")})
                                                .custom_parser(|s| {
                                                    let p = s.parse::<f64>();
                                                    match p { Ok(v) => Some(v / 100.0), _ => None,}
                                                })
                                            ).changed() { need_recalc = true; }
                                            ui.end_row();
                                        });
                                    };
                                });
                            });
                        });
                    };

                    // TUR
                    if self.columns.tur {
                        row.col(|ui| {
                            ui.label(format!("{:.2}", self.quantities[idx].tur));
                        });
                    };

                    // PFA
                    if self.columns.pfa {
                        row.col(|ui| {
                            ui.label(format!("{:.3} %", self.quantities[idx].pfa*100.0));
                        });
                    };
                    // CPFA
                    if self.columns.cpfa {
                        row.col(|ui| {
                            ui.label(format!("{:.3} %", self.quantities[idx].cpfa*100.0));
                        });
                    };
                    // PFR
                    if self.columns.pfr {
                        row.col(|ui| {
                            ui.label(format!("{:.3} %", self.quantities[idx].pfr*100.0));
                        });
                    };
                });
            });
        if need_recalc {
            self.recalc();  // SINGLE item
        }
    }
    fn draw_output(&mut self, ui: &mut egui::Ui) {
        if self.show_output {
            egui::ScrollArea::vertical()
            .id_salt("output")
            .min_scrolled_height(64.0)
            .show(ui, |ui| {
                ui.add(egui::TextEdit::multiline(&mut self.output.clone())
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                );
                });
        };
        if self.show_plots {
            self.draw_plots(ui);
        };
    }


    fn draw_plots(&mut self, ui: &mut egui::Ui) {
        let mut need_replot = false;

        ui.horizontal(|ui| {
            egui::ComboBox::new(ui.next_auto_id(), "")
            .selected_text(format!("{:?}", self.plot.plot_type))
            .show_ui(ui, |ui| {
                if ui.selectable_value(&mut self.plot.plot_type, PlotType::Reliability, "Reliability PDFs").clicked() { need_replot = true; };
                if ui.selectable_value(&mut self.plot.plot_type, PlotType::Utility, "Utility Curve").clicked() { need_replot = true; };
                if ui.selectable_value(&mut self.plot.plot_type, PlotType::Decay, "Reliability Decay").clicked() { need_replot = true; };
            });

            egui::ComboBox::new(ui.next_auto_id(), "")
                .selected_text(format!("{}", self.model.quantity[self.plot.qty].symbol))
                .show_ui(ui, |ui| {
                    for (i, qty) in self.model.quantity.iter().enumerate() {
                        if ui.selectable_value(&mut self.plot.qty, i, qty.symbol.clone()).clicked() {need_replot = true;}
                    }
            });
        });

        if need_replot {
            self.replot();
        }

        let (xlabel, ylabel) = match self.plot.plot_type {
            PlotType::Reliability => ("Indicated Value", "Reliability PDF"),
            PlotType::Utility => ("Indicated Value", "Probability of Successful Outcome %"),
            PlotType::Decay => ("Time since Calibration", "Reliability %"),
        };

        let c1: Color32 = Color32::from_hex("#888888").unwrap();

        Plot::new("my_plot")
        .x_axis_label(xlabel)
        .y_axis_label(ylabel)
        .show_grid(false)
        .legend(Legend::default())
        .show(ui, |plot_ui| {

            // Vertical Lines
            plot_ui.vline(VLine::new(self.plot.vline1).color(c1).style(LineStyle::Dashed{length: 4.0}));
            plot_ui.vline(VLine::new(self.plot.vline2).color(c1).style(LineStyle::Dashed{length: 4.0}));

            // Curves
            match self.plot.plot_type {
                PlotType::Reliability => {
                    plot_ui.line(Line::new(PlotPoints::new(self.plot.data1.clone())).name("BOP Reliability"));
                    plot_ui.line(Line::new(PlotPoints::new(self.plot.data2.clone())).name("EOP Reliability"));
                },
                PlotType::Utility => {
                    plot_ui.line(Line::new(PlotPoints::new(self.plot.data1.clone())).name("Utility Curve"));
                },
                PlotType::Decay => {
                    plot_ui.line(Line::new(PlotPoints::new(self.plot.data1.clone())).name("Reliability"));
                }
            }

        });

    }


    fn replot(&mut self) {
        // Calculate curves to plot and store in self

        let qty = &self.model.quantity[self.plot.qty];
        match self.plot.plot_type {
            PlotType::Reliability => {
                self.plot.vline1 = qty.utility.as_ref().unwrap().tolerance.low;
                self.plot.vline2 = qty.utility.as_ref().unwrap().tolerance.high;

                let n = 250;
                let width = self.plot.vline2 - self.plot.vline1;
                let low = self.plot.vline1 - width/2.0;
                let high = self.plot.vline2 + width/2.0;
                let step = (high - low) / n as f64;

                let mut bop = Vec::<[f64; 2]>::new();
                if let Some(bop_dist) = &self.quantities[self.plot.qty].bop_dist {
                    for i in 0..n {
                        let x = low + i as f64 * step;
                        bop.push((x, bop_dist.pdf(x)).into());
                    }
                }
                self.plot.data1 = bop;

                let mut eop = Vec::<[f64; 2]>::new();
                if let Some(eop_dist) = &self.quantities[self.plot.qty].eop_dist {
                    for i in 0..n {
                        let x = low + i as f64 * step;
                        eop.push((x, eop_dist.pdf(x)).into());
                    }
                }
                self.plot.data2 = eop;

            },
            PlotType::Utility => {
                self.plot.data2 = vec![];
                self.plot.vline1 = qty.utility.as_ref().unwrap().tolerance.low;
                self.plot.vline2 = qty.utility.as_ref().unwrap().tolerance.high;

                let n = 250;

                let mut util = Vec::<[f64; 2]>::new();
                if let Some(utility) = &self.quantities[self.plot.qty].util_dist {
                    let (low, high) = utility.domain();
                    let step = (high - low) / n as f64;

                    for i in 0..n {
                        let x = low + i as f64 * step;
                        util.push((x, utility.pdf(x)).into());
                    }
                }
                self.plot.data1 = util;
            },
            PlotType::Decay => {
                self.plot.data2 = vec![];
                let org_interval = if let Some(interval) = &qty.interval { interval.years } else { f64::NAN };
                self.plot.vline1 = org_interval;

                let n = 250;

                let mut dec = Vec::<[f64; 2]>::new();
                if let Some(decay) = &self.quantities[self.plot.qty].decay {
                    let test_interval = self.quantities[self.plot.qty].calc_interval;
                    let high = test_interval.max(org_interval) * 1.25;
                    let step = high / n as f64;

                    self.plot.vline2 = test_interval;

                    for i in 0..n {
                        let x = i as f64 * step;
                        dec.push((x, decay.reliability_time(x)*100.0).into());
                    }
                }
                self.plot.data1 = dec;
            },
        };
    }
}


enum ToleranceMode {
    PlusMinus,
    Greater,
    Less,
    Range,
}

struct ToleranceUi {
    mode: ToleranceMode,
    value1: f64,
    value2: f64,
    enable: bool,
}
impl ToleranceUi {
    fn new(enable: bool) -> Self {
        Self{
            mode: ToleranceMode::PlusMinus,
            value1: 0.0,
            value2: 1.0,
            enable: enable,
        }
    }
    fn to_string(&self) -> String {
        match self.mode {
            ToleranceMode::PlusMinus => format!("{} ± {:.4}", self.value1, self.value2),
            ToleranceMode::Less => format!("< {}", self.value1),
            ToleranceMode::Greater => format!("> {}", self.value1),
            ToleranceMode::Range => format!("{} ↔ {}", self.value1, self.value2),
        }
    }
    fn from_tol(tol: &Tolerance) -> Self {
        let center = (tol.high + tol.low) / 2.0;
        if !tol.high.is_finite() {
            ToleranceUi{
                mode: ToleranceMode::Greater,
                value1: tol.low,
                value2: f64::NAN,
                enable: true,
            }
        } else if !tol.low.is_finite() {
            ToleranceUi{
                mode: ToleranceMode::Less,
                value1: tol.high,
                value2: f64::NAN,
                enable: true,
            }
        } else if ((tol.high - center) - (center - tol.low)).abs() < 1E-15 {
            ToleranceUi{
                mode: ToleranceMode::PlusMinus,
                value1: center,
                value2: tol.high - center,
                enable: true,
            }
        } else {
            ToleranceUi{
                mode: ToleranceMode::Range,
                value1: tol.low,
                value2: tol.high,
                enable: true,
            }
        }
    }
    fn to_tol(&self) -> Option<Tolerance> {
        if !self.enable {
            None
        } else {
            match self.mode {
                ToleranceMode::PlusMinus => {
                    Some(Tolerance{
                            low: self.value1 - self.value2,
                            high: self.value1 + self.value2
                        })
                },
                ToleranceMode::Greater => {
                    Some(Tolerance{low: self.value1, high: f64::INFINITY})
                },
                ToleranceMode::Less => {
                    Some(Tolerance{low: -f64::INFINITY, high: self.value1})
                },
                ToleranceMode::Range => {
                    Some(Tolerance{low: self.value1, high: self.value2})
                }
            }
        }
    }

}


fn tolerance_widget(ui: &mut egui::Ui, tol: &mut ToleranceUi) -> bool {
    let mut need_recalc = false;
    match tol.mode {
        ToleranceMode::PlusMinus => {
            ui.horizontal(|ui| {
                if ui.add(egui::DragValue::new(&mut tol.value1).speed(0.01)).changed() {
                    need_recalc = true;
                };
                if ui.button("±").clicked() {
                    tol.mode = ToleranceMode::Greater;
                    need_recalc = true;
                };
                if ui.add(egui::DragValue::new(&mut tol.value2).speed(0.01)).changed() {
                    need_recalc = true;
                };
            });
        },
        ToleranceMode::Greater => {
            ui.horizontal(|ui| {
                if ui.button(">").clicked() {
                    tol.mode = ToleranceMode::Less;
                    need_recalc = true;
                };
                if ui.add(egui::DragValue::new(&mut tol.value1).speed(0.01)).changed() {
                    need_recalc = true;
                };
            });
        },
        ToleranceMode::Less => {
            ui.horizontal(|ui| {
                if ui.button("<").clicked() {
                    tol.mode = ToleranceMode::Range;
                    need_recalc = true;
                };
                if ui.add(egui::DragValue::new(&mut tol.value1).speed(0.01)).changed() {
                    need_recalc = true;
                };
            });
        },
        ToleranceMode::Range => {
            ui.horizontal(|ui| {
                if ui.add(egui::DragValue::new(&mut tol.value1).speed(0.01)).changed() {
                    need_recalc = true;
                };
                if ui.button("↔").clicked() {
                    tol.mode = ToleranceMode::PlusMinus;
                    need_recalc = true;
                };
                if ui.add(egui::DragValue::new(&mut tol.value2).speed(0.01)).changed() {
                    need_recalc = true;
                };
            });
        },
    }
    need_recalc
}
