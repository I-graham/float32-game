use winit::event;
use winit::event::Event;

mod render;
use render::*;

enum Stage {
	Playing,
}

struct GameState {
	stage : Stage,
}

impl GameState {
	fn new() -> Self {
		Self {
			stage : Stage::Playing,
		}
	}
}

fn draw(renderer : &mut Renderer, state : &GameState) {

	let frame = renderer.swap.get_next_texture().expect("No Texture!");

	let mut encoder = renderer.device.create_command_encoder(
		&wgpu::CommandEncoderDescriptor {
			label: Some("draw"),
		}
	);

	let render_pass = encoder.begin_render_pass(
		&wgpu::RenderPassDescriptor {
			color_attachments : &[

				wgpu::RenderPassColorAttachmentDescriptor {

					attachment : &frame.view,
					resolve_target : None,
					load_op : wgpu::LoadOp::Clear,
					store_op : wgpu::StoreOp::Store,
					clear_color : wgpu::Color {
						r : 0.1,
						g : 0.2,
						b : 0.3,
						a : 1.0,
					},

				},
			],

			depth_stencil_attachment : None,
				
		}
	);

	drop(render_pass);

	renderer.queue.submit(&[encoder.finish()]);

}

fn main() {
	let event_loop = winit::event_loop::EventLoop::new();

	let window = winit::window::WindowBuilder::new()
					.with_title("Float32")
					.build(&event_loop)
					.unwrap();

	let mut renderer = Renderer::new(&window);

	let state = GameState::new();

	let 

	event_loop.run(move |event, _, control_flow| {


		match event {

			Event::WindowEvent {
				event,
				..
			} => {
				match event {

					event::WindowEvent::CloseRequested => *control_flow = winit::event_loop::ControlFlow::Exit,

					event::WindowEvent::ScaleFactorChanged { new_inner_size : dims, .. } => renderer.resize(*dims),

					event::WindowEvent::Resized(dims) if (dims.height > 0) && (dims.width > 0) => renderer.resize(dims),
					event::WindowEvent::Resized(_) => window.set_inner_size(renderer.win_size),

					_ => {},
					
				}
			},

			Event::MainEventsCleared =>  { window.request_redraw(); },

			Event::RedrawRequested(_) => draw(&mut renderer, &state),

			_ => (),

		};

	});

}
