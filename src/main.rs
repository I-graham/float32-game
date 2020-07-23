use winit::event;
use winit::event::Event;

use std::collections::HashMap;

mod render;
use render::*;

const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.0,
	0.0, 0.0, 0.5, 1.0,
);

const START_SIZE : winit::dpi::LogicalSize<f32> = winit::dpi::LogicalSize {
	width : 800.0,
	height : 600.0,
}; 

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

struct Instance {
	position: cgmath::Vector3<f32>,
	rotation: cgmath::Quaternion<f32>,
}

impl Instance {
	fn to_matrix(&self) -> cgmath::Matrix4<f32> {
		cgmath::Matrix4::from_translation(self.position)
			* cgmath::Matrix4::from(self.rotation)
	}
}

struct Model {
	mesh : (wgpu::Buffer, wgpu::Buffer, u32, u32),
	instances : Vec<cgmath::Matrix4<f32>>,
	visible : bool,
}

impl Model {
	fn new(device : &wgpu::Device, vertices : &[Vertex], indices : &[u32]) -> Self {
		let mesh = {
			let vert_buff = device.create_buffer_with_data(
				render::to_char_slice(vertices),
				wgpu::BufferUsage::VERTEX,
			);

			let ind_buff = device.create_buffer_with_data(
				render::to_char_slice(indices),
				wgpu::BufferUsage::INDEX,
			);

			(vert_buff, ind_buff, vertices.len() as u32, indices.len() as u32)
		};

		let instances = vec![];

		Self {
			mesh,
			instances,
			visible : true,
		}

	}
}

struct Scene {
	objects : HashMap<&'static str, Model>,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
struct Uniform {
	time : i32,
	__align0 : [i32; 3],
	cam_position : [f32; 3],
	__align1 : [i32; 1],
	cam_proj : cgmath::Matrix4<f32>,
	light_position : [f32; 3],
	__align2 : [i32; 1],
	light_color : [f32; 3],
	__align3 : [i32; 1],
}

impl Default for Uniform {
	fn default() -> Self {
		use cgmath::prelude::SquareMatrix;
		Uniform {
			time : 0,
			cam_position : [0.0,0.0,0.0],
			cam_proj : cgmath::Matrix4::identity(),
			light_position : [0.0, 0.0, 0.0],
			light_color : [1.0, 1.0, 1.0],
			__align0 : [0; 3],
			__align1 : [0; 1],
			__align2 : [0; 1],
			__align3 : [0; 1]
		}
	}
}

struct WinState {
	mouse_pos    : (f64, f64),
	mouse_down_l : bool,
	win_size     : winit::dpi::PhysicalSize<u32>,
	w_pressed    : bool,
	a_pressed    : bool,
	s_pressed    : bool,
	d_pressed    : bool,
	shft_down    : bool,
	ctrl_down    : bool, 
}

impl WinState {
	pub fn new(renderer : &render::Renderer) -> Self {
		Self {
			win_size     : renderer.win_size,
			mouse_pos    : (0.0, 0.0),
			mouse_down_l : false,
			w_pressed    : false,
			a_pressed    : false,
			s_pressed    : false,
			d_pressed    : false,
			shft_down    : false,
			ctrl_down    : false,
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
	position : [f32; 3],
	color : [f32; 3],
	normal : [f32; 3],
}

impl Vertex {
	
	fn load_mesh(filename : &str, color : [f32; 3], scale : f32) -> (Vec<Vertex>, Vec<u32>) {
		let obj = tobj::load_obj(filename, true).expect(format!("{}{}{}\n", "Missing asset :'", filename, "'").as_str());
		
		let (mut models, _materials) = obj;

		let model = models.remove(0);

		let mesh = model.mesh;
		
		let indices = mesh.indices;
		
		let length = mesh.positions.len() / 3;

		let mut verts : Vec<Vertex> = Vec::with_capacity(length);

		let mut position_iter = mesh.positions.iter().map(|x| scale * x);

		let mut normals_iter = mesh.normals.iter().map(|x| *x);

		for _ in 0..length {

			verts.push(
				Vertex {
					position : [position_iter.next().unwrap(), position_iter.next().unwrap(), position_iter.next().unwrap()],
					color,
					normal    : [normals_iter.next().unwrap(), normals_iter.next().unwrap(), normals_iter.next().unwrap()],
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
		wgpu::VertexAttributeDescriptor {
			offset : std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
			shader_location : 2,
			format : wgpu::VertexFormat::Float3,
		},
	],
};

enum Stage {
	Menu,	
	Playing,
}

struct Boat {
	position : Instance,
}

struct GameState {
	stage : Stage,
	win_state : WinState,
	uniforms : Uniform,
	camera : Camera,
	renderer : Renderer,
	scene : Scene,

	boats : Vec<Boat>,
	
}

impl GameState {

	fn draw(&mut self) {

		match self.stage {
			Stage::Menu{..} => self.draw_menu(),
			Stage::Playing{..} => self.draw_game(),
		};
		
	}
	
	fn draw_menu(&mut self){
		
		if let Stage::Menu = self.stage {
			
			let frame = self.renderer.swap.get_next_texture().unwrap();
			
			{

				let num_boats = self.boats.len();
				
				let boat_model = self.scene.objects.get_mut("boat").unwrap();

				boat_model.instances.reserve(num_boats);

				boat_model.instances.truncate(0);

				boat_model.instances.extend(self.boats.iter().map(|boat| boat.position.to_matrix()));

			}
				
			let mut encoder = self.renderer.begin();

			{
				let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
						attachment : &frame.view,
						resolve_target : None,
						load_op: wgpu::LoadOp::Clear,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.0,
							g: 0.8,
							b: 1.0,
							a: 1.0,
						},
						}],
					depth_stencil_attachment: Some( wgpu::RenderPassDepthStencilAttachmentDescriptor {
						attachment : &self.renderer.depth_buffer.1,
						depth_load_op : wgpu::LoadOp::Clear,
						depth_store_op : wgpu::StoreOp::Store,
						clear_depth : 1.0,
						stencil_load_op : wgpu::LoadOp::Clear,
						stencil_store_op : wgpu::StoreOp::Store,
						clear_stencil : 0,
					}),
				});

				render_pass.set_pipeline(&self.renderer.pipeline);

				render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

				render_pass.draw(0..0, 0..1);

			}

			for model in self.scene.objects.values() {

				if !model.visible {
					continue
				}

				let instances = &model.instances;

				let instance_buf = self.renderer.device.create_buffer_with_data(
					render::to_char_slice(instances.as_slice()),
					wgpu::BufferUsage::COPY_SRC
				);

				encoder.copy_buffer_to_buffer(
					&instance_buf,
					0 as wgpu::BufferAddress,
					&self.renderer.instance_buf,
					0 as wgpu::BufferAddress,
					render::Renderer::INSTANCE_SIZE,
				);

				let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
						attachment : &frame.view,
						resolve_target : None,
						load_op: wgpu::LoadOp::Load,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.0,
							g: 0.8,
							b: 1.0,
							a: 1.0,
						},
						}],
					depth_stencil_attachment: Some( wgpu::RenderPassDepthStencilAttachmentDescriptor {
						attachment : &self.renderer.depth_buffer.1,
						depth_load_op : wgpu::LoadOp::Load,
						depth_store_op : wgpu::StoreOp::Store,
						clear_depth : 1.0,
						stencil_load_op : wgpu::LoadOp::Clear,
						stencil_store_op : wgpu::StoreOp::Store,
						clear_stencil : 0,
					}),
				});

				render_pass.set_pipeline(&self.renderer.pipeline);
				
				render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

				let mesh = &model.mesh;

				render_pass.set_vertex_buffer(0, &mesh.0, 0, 0);

				render_pass.set_index_buffer(&mesh.1, 0, 0);

				render_pass.draw_indexed(0..mesh.3, 0, 0..instances.len() as u32);

				drop(render_pass);
			}

			use wgpu_glyph::{ab_glyph, GlyphBrushBuilder, Section, Text, Layout, HorizontalAlign};

			let font = ab_glyph::FontArc::try_from_slice(include_bytes!("../data/bahnschrift.ttf"))
								.expect("unable to load font.");

			let mut brush = GlyphBrushBuilder::using_font(font).build(&self.renderer.device, self.renderer.sc_desc.format);

			let section = Section {
				screen_position: (self.win_state.win_size.width as f32 / 2.0f32, 1.8 * self.win_state.win_size.height as f32 / 2.0f32),
				text : vec![Text::new("Press space to begin").with_scale(70.0).with_color([0.0,0.0,0.0,1.0])],
				layout : Layout::default_single_line().h_align(HorizontalAlign::Center),
				..Section::default()
			};

			brush.queue(section);

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

		if let Stage::Playing = &self.stage {
			
			let frame = self.renderer.swap.get_next_texture().unwrap();

			let mut encoder = self.renderer.begin();

			{
				let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
						attachment : &frame.view,
						resolve_target : None,
						load_op: wgpu::LoadOp::Clear,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.0,
							g: 0.8,
							b: 1.0,
							a: 1.0,
						},
						}],
					depth_stencil_attachment: Some( wgpu::RenderPassDepthStencilAttachmentDescriptor {
						attachment : &self.renderer.depth_buffer.1,
						depth_load_op : wgpu::LoadOp::Clear,
						depth_store_op : wgpu::StoreOp::Store,
						clear_depth : 1.0,
						stencil_load_op : wgpu::LoadOp::Clear,
						stencil_store_op : wgpu::StoreOp::Store,
						clear_stencil : 0,
					}),
				});

				render_pass.set_pipeline(&self.renderer.pipeline);

				render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

				render_pass.draw(0..0, 0..1);

			}

			for model in self.scene.objects.values() {

				let instances = &model.instances;

				let instance_buf = self.renderer.device.create_buffer_with_data(
					render::to_char_slice(instances.as_slice()),
					wgpu::BufferUsage::COPY_SRC
				);

				encoder.copy_buffer_to_buffer(
					&instance_buf,
					0 as wgpu::BufferAddress,
					&self.renderer.instance_buf,
					0 as wgpu::BufferAddress,
					render::Renderer::INSTANCE_SIZE,
				);

				let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					color_attachments : &[wgpu::RenderPassColorAttachmentDescriptor {
						attachment : &frame.view,
						resolve_target : None,
						load_op: wgpu::LoadOp::Load,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.0,
							g: 0.8,
							b: 1.0,
							a: 1.0,
						},
						}],
					depth_stencil_attachment: Some( wgpu::RenderPassDepthStencilAttachmentDescriptor {
						attachment : &self.renderer.depth_buffer.1,
						depth_load_op : wgpu::LoadOp::Load,
						depth_store_op : wgpu::StoreOp::Store,
						clear_depth : 1.0,
						stencil_load_op : wgpu::LoadOp::Clear,
						stencil_store_op : wgpu::StoreOp::Store,
						clear_stencil : 0,
					}),
				});

				render_pass.set_pipeline(&self.renderer.pipeline);
				
				render_pass.set_bind_group(0, &self.renderer.uniform_bg, &[]);

				let mesh = &model.mesh;

				render_pass.set_vertex_buffer(0, &mesh.0, 0, 0);

				render_pass.set_index_buffer(&mesh.1, 0, 0);

				render_pass.draw_indexed(0..mesh.3, 0, 0..instances.len() as u32);

				drop(render_pass);
			}
		
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

		self.uniforms.cam_position = self.camera.eye.into();

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

		self.uniforms.light_position = self.uniforms.cam_position;

		self.camera.target = self.camera.eye - cgmath::Vector3::unit_z();
		
		self.uniforms.cam_proj = self.camera.build_view_projection_matrix();

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
			
			#[allow(deprecated)]
			event::WindowEvent::KeyboardInput {
				input : event::KeyboardInput {
					virtual_keycode : Some(key),
					state,
					modifiers,
					..
				},
				..
			} => {

				self.win_state.ctrl_down = modifiers.ctrl();
				self.win_state.shft_down = modifiers.shift();

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
						
						if let Stage::Menu = &self.stage {
							self.stage = Stage::Playing;
							setup_game_models(&self.renderer.device, &mut self.scene);
						};

					},

					_ => {},
				}

			},

			event::WindowEvent::MouseInput {
				button,
				state,
				..
			} => {
				match button {

					event::MouseButton::Left => {
						self.win_state.mouse_down_l = *state == event::ElementState::Pressed;
					},

					_ => (),
				}
			},

			_ => (),

		};
	}
}

fn setup_menu_models(device : &wgpu::Device, scene : &mut Scene) {

	let (verts, indices) = Vertex::load_mesh("data/floater.obj", [0.8,0.8,0.8], 0.5);

	let vertices : &[Vertex] = verts.as_slice();

	let indices : &[u32] = indices.as_slice();

	scene.objects.insert("boat", Model::new(device, vertices, indices));
	
	let model = scene.objects.get_mut("boat").unwrap();
	
	model.instances.push(
		cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0,0.0,-1.0)) * cgmath::Matrix4::from_scale(0.5)
	);

}

fn setup_game_models(device : &wgpu::Device, scene : &mut Scene) {

}

async fn entry(event_loop : winit::event_loop::EventLoop<()>, window : winit::window::Window) {
	
	let camera = Camera {
		eye    : (0.0, 0.0, 1.0).into(),
		target : (0.0, 0.0, 0.0).into(),
		up     : cgmath::Vector3::unit_y(),
		aspect : window.inner_size().width as f32 / window.inner_size().height as f32,
		fovy   : 45.0,
		znear  : 0.01,
		zfar   : 100.0,
	};

	let uniforms = Uniform {
		cam_proj : camera.build_view_projection_matrix(),
		time : 0,
		cam_position : camera.eye.into(),
		..Default::default()
	};	

	let renderer = Renderer::new(&window, &[VERTEX_DESC], &[uniforms]).await;
	
	let mut state = {

		let mut scene = Scene {
			objects : HashMap::new(),
		};

		setup_menu_models(&renderer.device, &mut scene);
		
		let mut boats = vec![];

		use cgmath::prelude::Rotation3;

		boats.push( Boat {
			position : Instance {
				position : [0.0, 0.0, -1.0].into(),
				rotation : cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(0.0)),
			}
		});

		GameState {
			stage : Stage::Menu,
			win_state : WinState::new(&renderer),
			uniforms,
			camera,
			renderer,
			scene,
			boats,
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

					event::WindowEvent::Resized(dims) => {
						
						state.renderer.resize(*dims);
						state.win_state.win_size = *dims;
					},

					event::WindowEvent::KeyboardInput { input : event::KeyboardInput {
						virtual_keycode : Some(key),
						state,
						..
					}, .. } => {
						match key {

							event::VirtualKeyCode::F11 if *state == event::ElementState::Released => {
								if let Some(_) = window.fullscreen() {
									window.set_fullscreen(None);
								} else {
									window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(window.primary_monitor())));
								}
							},

							event::VirtualKeyCode::Escape => {
								*control_flow = winit::event_loop::ControlFlow::Exit;
							},

							_ => {},

						}
					},

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
		.with_min_inner_size(START_SIZE)
		.with_title("Float32")
		.with_inner_size(START_SIZE)
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