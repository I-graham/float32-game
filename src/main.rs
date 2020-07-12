use winit::event;
use winit::event::Event;

mod gui;
use gui::*;

mod render;
use render::*;

pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
	pub eye: cgmath::Point3<f32>,
	pub target: cgmath::Point3<f32>,
	pub up: cgmath::Vector3<f32>,
	pub aspect: f32,
	pub fovy: f32,
	pub znear: f32,
	pub zfar: f32,
}

impl Camera {
	fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
		
		let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
		
		let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

		return OPENGL_TO_WGPU_MATRIX * proj * view;
	}
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct Uniform {
	proj : cgmath::Matrix4<f32>,
	time : i32,
}

struct WinState {
	mouse_pos : (f64, f64),
	win_size  : winit::dpi::PhysicalSize<u32>,
	w_pressed : bool,
	a_pressed : bool,
	s_pressed : bool,
	d_pressed : bool,
	shft_down : bool,
	ctrl_down : bool, 
}

impl WinState {
	pub fn new(renderer : &render::Renderer) -> Self {
		Self {
			mouse_pos : (0.0, 0.0),
			win_size  : renderer.win_size,
			w_pressed : false,
			a_pressed : false,
			s_pressed : false,
			d_pressed : false,
			shft_down : false,
			ctrl_down : false,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
	position: [f32; 3],
	color: [f32; 3],	
}

impl Vertex {
	
	fn load_mesh(filename : &str, scale : f32) -> (Vec<Vertex>, Vec<u32>) {
		let obj = tobj::load_obj(filename, true).expect(format!("{}{}{}\n", "Missing asset :'", filename, "'").as_str());
		
		let (mut models, _materials) = obj;

		let model = models.remove(0);

		let mesh = model.mesh;
		
		let indices = mesh.indices;
		
		let length = mesh.positions.len();

		let mut verts : Vec<Vertex> = Vec::with_capacity(length);

		let mut position_iter = mesh.positions.iter().map(|x| scale * x);
		let normals = mesh.normals;

		for i in 0..length / 3 {
			verts.push(
				Vertex {
					position : [position_iter.next().unwrap(), position_iter.next().unwrap(), position_iter.next().unwrap()],
					color    : [normals[i / 3], normals[i / 3 + 1], normals[i / 3 + 2]],
				}
			);
		}

		(verts, indices)

	}

}

const VERTEX_DESC : wgpu::VertexBufferDescriptor = wgpu::VertexBufferDescriptor {
	stride : std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
	step_mode : wgpu::InputStepMode::Vertex,
	attributes : &[
		wgpu::VertexAttributeDescriptor {
			offset : 0,
			shader_location : 0,
			format : wgpu::VertexFormat::Float3,
		},
		wgpu::VertexAttributeDescriptor {
			offset : std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
			shader_location : 1,
			format : wgpu::VertexFormat::Float3,
		},
	],
};

enum Stage {
	Menu {
		vertices : (wgpu::Buffer, wgpu::Buffer, u32, u32),
	},
	
	Playing {
		vertices : (wgpu::Buffer, wgpu::Buffer, u32, u32),		
	},
}

struct GameState {
	stage : Stage,
	win_state : WinState,
	uniforms : Uniform,
	camera : Camera,
	renderer : Renderer,
}

impl GameState {

	fn draw(&mut self) {

		match self.stage {
			Stage::Menu{..} => self.draw_menu(),
			Stage::Playing{..} => self.draw_game(),
		};

	}

	fn draw_menu(&mut self){

		if let Stage::Menu {
			vertices,
		} = &self.stage {
			
			let frame = self.renderer.swap.get_next_texture().unwrap();

			let mut encoder = self.renderer.begin();

			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments : &[ wgpu::RenderPassColorAttachmentDescriptor {
					attachment : &frame.view,
					resolve_target : None,
					load_op: wgpu::LoadOp::Clear,
					store_op: wgpu::StoreOp::Store,
					clear_color: wgpu::Color {
						r: 0.1,
						g: 0.2,
						b: 0.3,
						a: 1.0,
					},
					}],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.renderer.pipeline);
			
			render_pass.set_vertex_buffer(0, &vertices.0, 0, 0);

			render_pass.set_index_buffer(&vertices.1, 0, 0);
			
			render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

			render_pass.draw_indexed(0..vertices.3, 0, 0..1);
			
			use wgpu_glyph::{ab_glyph, GlyphBrushBuilder, Section, Text, Layout, HorizontalAlign};

			let font = ab_glyph::FontArc::try_from_slice(include_bytes!("../data/bahnschrift.ttf"))
								.expect("unable to load font.");

			let mut brush = GlyphBrushBuilder::using_font(font).build(&self.renderer.device, self.renderer.sc_desc.format);

			let section = Section {
				screen_position: (self.win_state.win_size.width as f32 / 2.0f32, 1.5 * self.win_state.win_size.height as f32 / 2.0f32),
				text : vec![Text::new("Press space to begin").with_scale(70.0)],
				layout : Layout::default_single_line().h_align(HorizontalAlign::Center),
				..Section::default()
			};

			brush.queue(section);

			drop(render_pass);

			brush.draw_queued(
				&self.renderer.device,
				&mut encoder,
				&frame.view,
				self.renderer.sc_desc.width,
				self.renderer.sc_desc.height,
			).unwrap();
		
			self.renderer.queue.submit(&[encoder.finish()]);
			
		} else  { panic!("type mismatch!"); }
	}
	
	fn draw_game(&mut self) {

		if let Stage::Playing {
			vertices,
		} = &self.stage {
			
			let frame = self.renderer.swap.get_next_texture().unwrap();

			let mut encoder = self.renderer.begin();

			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments : &[ wgpu::RenderPassColorAttachmentDescriptor {
					attachment : &frame.view,
					resolve_target : None,
					load_op: wgpu::LoadOp::Clear,
					store_op: wgpu::StoreOp::Store,
					clear_color: wgpu::Color {
						r: 0.1,
						g: 0.2,
						b: 0.3,
						a: 1.0,
					},
				}],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.renderer.pipeline);
			
			render_pass.set_vertex_buffer(0, &vertices.0, 0, 0);

			render_pass.set_index_buffer(&vertices.1, 0, 0);
			
			render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

			render_pass.draw_indexed(0..vertices.3, 0, 0..1);
			
			use wgpu_glyph::{ab_glyph, GlyphBrushBuilder, Section, Text, Layout, HorizontalAlign};

			let font = ab_glyph::FontArc::try_from_slice(include_bytes!("../data/bahnschrift.ttf"))
								.expect("unable to load font.");

			let mut brush = GlyphBrushBuilder::using_font(font).build(&self.renderer.device, self.renderer.sc_desc.format);

			let section = Section {
				screen_position: (self.win_state.win_size.width as f32 / 2.0f32, 1.5 * self.win_state.win_size.height as f32 / 2.0f32),
				text : vec![Text::new("Press space to begin").with_scale(70.0)],
				layout : Layout::default_single_line().h_align(HorizontalAlign::Center),
				..Section::default()
			};

			brush.queue(section);

			drop(render_pass);

			brush.draw_queued(
				&self.renderer.device,
				&mut encoder,
				&frame.view,
				self.renderer.sc_desc.width,
				self.renderer.sc_desc.height,
			).unwrap();
		
			self.renderer.queue.submit(&[encoder.finish()]);
			
		} else  { panic!("type mismatch!"); }

	}

	fn update(&mut self) {
		match self.stage {
			Stage::Menu {..} => self.update_menu(),
			Stage::Playing {..} => self.update_game(),
		}
	}

	fn update_menu(&mut self) {

		self.uniforms.time += 1;

		self.upload_uniform();

	}	

	fn update_game(&mut self) {
		self.uniforms.time += 1;

		const SPEED : f32 = 0.05;

		self.camera.eye += cgmath::Vector3::new(
			(self.win_state.d_pressed as i8 - self.win_state.a_pressed as i8) as f32,
			(self.win_state.w_pressed as i8 - self.win_state.s_pressed as i8) as f32,
			(self.win_state.ctrl_down as i8 - self.win_state.shft_down as i8) as f32,
		) * SPEED;

		self.camera.target = self.camera.eye - cgmath::Vector3::unit_z();
		self.uniforms.proj = self.camera.build_view_projection_matrix();

		self.upload_uniform();

	}

	fn upload_uniform(&self) {
		let mut encoder = self.renderer.begin();
		
		let new_uniform = self.renderer.device.create_buffer_with_data(
			render::to_char_slice(&[self.uniforms]),
			wgpu::BufferUsage::COPY_SRC,
		);

		encoder.copy_buffer_to_buffer(
			&new_uniform,
			0 as wgpu::BufferAddress,
			&self.renderer.uniform_buf,
			0 as wgpu::BufferAddress,
			std::mem::size_of::<Uniform>() as wgpu::BufferAddress,
		);

		self.renderer.queue.submit(&[encoder.finish()]);

	}

	fn input(&mut self, event : &winit::event::WindowEvent) {
		match event {
			event::WindowEvent::CursorMoved {
				position,
				..
			} => { self.win_state.mouse_pos = ((2.0 * position.x / self.win_state.win_size.width as f64) - 1.0, (-2.0 * position.y / self.win_state.win_size.height as f64) + 1.0); },

			event::WindowEvent::KeyboardInput {
				input : event::KeyboardInput {
					virtual_keycode : Some(key),
					state,
					modifiers,
					..
				},
				..
			} => {

				match key {

					event::VirtualKeyCode::W => {
						self.win_state.w_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::A => {
						self.win_state.a_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::S => {
						self.win_state.s_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::D => {
						self.win_state.d_pressed = *state == event::ElementState::Pressed;
					},

					event::VirtualKeyCode::Space if *state == event::ElementState::Pressed => {
						
						if let Stage::Menu {
							vertices : (vert, ind, vert_len, ind_len)
						} = &self.stage {

							let vert_size = *vert_len as u64 * std::mem::size_of::<Vertex>() as wgpu::BufferAddress;
							let ind_size  = *ind_len as u64 * std::mem::size_of::<Vertex>() as wgpu::BufferAddress;

							let new_vert = self.renderer.device.create_buffer(&wgpu::BufferDescriptor {
								label : None,
								size  : vert_size,
								usage : wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
							});
							
							let new_ind = self.renderer.device.create_buffer(&wgpu::BufferDescriptor {
								label : None,
								size  : ind_size,
								usage : wgpu::BufferUsage::INDEX | wgpu::BufferUsage::COPY_DST,
							});

							let mut encoder = self.renderer.begin();

							encoder.copy_buffer_to_buffer(
								vert,
								0 as wgpu::BufferAddress,
								&new_vert,
								0 as wgpu::BufferAddress,
								vert_size
							);

							encoder.copy_buffer_to_buffer(
								ind,
								0 as wgpu::BufferAddress,
								&new_ind,
								0 as wgpu::BufferAddress,
								ind_size
							);

							self.renderer.queue.submit(&[encoder.finish()]);

							self.stage = Stage::Playing { 
								 vertices : (new_vert, new_ind, *vert_len, *ind_len),
							};
						};

					},

					_ => {
						self.win_state.ctrl_down = modifiers.ctrl();
						self.win_state.shft_down = modifiers.shift();
					},
				}

			},

			_ => (),

		};
	}
}

async fn entry(event_loop : winit::event_loop::EventLoop<()>, window : winit::window::Window) {
	
	use cgmath::prelude::SquareMatrix;

	let uniforms = Uniform {
		proj : cgmath::Matrix4::identity(),
		time : 0,
	};

	let renderer = Renderer::new(&window, &[VERTEX_DESC], &[uniforms]).await;

	let camera = Camera {
		eye    : (0.0, 0.0, 3.0).into(),
		target : (0.0, 0.0, 0.0).into(),
		up     : cgmath::Vector3::unit_y(),
		aspect : renderer.sc_desc.width as f32 / renderer.sc_desc.height as f32,
		fovy   : 45.0,
		znear  : 0.1,
		zfar   : 100.0,
	};

	let mut state = {

		let (verts, indices) = Vertex::load_mesh("data/floater.obj", 0.5);

		let vertices : &[Vertex] = verts.as_slice();

		let indices : &[u32] = indices.as_slice();

		let vertex_buff = renderer.device.create_buffer_with_data(
			render::to_char_slice(vertices),
			wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_SRC,
		);

		let index_buff = renderer.device.create_buffer_with_data(
			render::to_char_slice(indices),
			wgpu::BufferUsage::INDEX | wgpu::BufferUsage::COPY_SRC,
		);


		GameState {
			stage : Stage::Menu {
				vertices : (vertex_buff, index_buff, vertices.len() as u32, indices.len() as u32),
			},
			win_state : WinState::new(&renderer),
			uniforms,
			camera,
			renderer,
		}
	};

	let mut fps = 0;
	let mut timer = std::time::Instant::now();

	event_loop.run(move |event, _, control_flow| {

		match event {

			Event::WindowEvent {
				event,
				window_id
			} if window_id == window.id() => {
				match &event {

					event::WindowEvent::CloseRequested => *control_flow = winit::event_loop::ControlFlow::Exit,

					event::WindowEvent::ScaleFactorChanged { new_inner_size : dims, .. } => state.renderer.resize(**dims),

					event::WindowEvent::Resized(dims) if (dims.height > 0) && (dims.width > 0) => { state.renderer.resize(*dims); state.win_state.win_size = *dims;},
					event::WindowEvent::Resized(_) => window.set_inner_size(state.renderer.win_size),

					_ => {},
					
				}
				
				state.input(&event);
			},

			Event::MainEventsCleared =>  {
				
				state.update();

				window.request_redraw();
				fps += 1;

				let now = std::time::Instant::now();

				if now.duration_since(timer).as_secs() >= 1 {
					println!("fps : {}", fps);
					fps = 0;
					timer = now;
				}

			},

			Event::RedrawRequested(_) => state.draw(),

			_ => (),

		};

	});

}

fn main() {

	let event_loop = winit::event_loop::EventLoop::new();
	
	let window = winit::window::WindowBuilder::new()
		.with_title("Float32")
		.with_inner_size(winit::dpi::LogicalSize { width: 800, height: 600})
		.build(&event_loop)
		.unwrap();

	window.set_cursor_icon(winit::window::CursorIcon::Default);

	#[cfg(target_arch = "wasm32")]
	{
		std::panic::set_hook(Box::new(console_error_panic_hook::hook));

		web_sys::window()
			.and_then(|win | win.document())
			.and_then(|doc | doc.document())
			.and_then(|body| body.append_child(&web_sys::Element::from(window.canvas())).ok()).expect("unable to create canvas to document body");

		wasm_bindgen_futures::spawn_local(entry(event_loop, window));
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		futures::executor::block_on(entry(event_loop, window));
	}	
}