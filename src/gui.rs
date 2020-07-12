

struct Button {
	pub area : (std::ops::Range<f64>, std::ops::Range<f64>),
	pub is_clicked : bool,
}

impl Button {
	pub fn reset(&mut self) -> bool {
		let ret = self.is_clicked;
		self.is_clicked = false;
		ret
	}

	pub fn check_clicked(&mut self, pos : &(f64, f64)) -> bool {
		self.is_clicked = self.area.0.contains(&pos.0) && self.area.1.contains(&pos.1);
		self.is_clicked
	}

}