pub struct Renderer {

	pub win_size : winit::dpi::PhysicalSize<u32>,
	
	pub surface  : wgpu::Surface,
	pub adapter  : wgpu::Adapter,
	pub device   : wgpu::Device,
	pub queue    : wgpu::Queue,
	pub sc_desc  : wgpu::SwapChainDescriptor,
	pub swap     : wgpu::SwapChain,


}

impl Renderer {
	pub fn new(win : &winit::window::Window) -> Self{

		let win_size = win.inner_size();

		let surface = wgpu::Surface::create(win);

		let adapter = futures::executor::block_on(wgpu::Adapter::request(
			&wgpu::RequestAdapterOptions {

				power_preference : wgpu::PowerPreference::HighPerformance,
				compatible_surface : Some(&surface),

			},
			
			wgpu::BackendBit::PRIMARY,
		
		)).unwrap();

		let (device, queue) = futures::executor::block_on(adapter.request_device(
			&wgpu::DeviceDescriptor {
				extensions: wgpu::Extensions {
					anisotropic_filtering: false,
				},
				limits : Default::default(),
			}
		));

		let sc_desc = wgpu::SwapChainDescriptor {

			usage        : wgpu::TextureUsage::OUTPUT_ATTACHMENT,
			format       : wgpu::TextureFormat::Bgra8UnormSrgb,
			width        : win_size.width,
			height       : win_size.height,
			present_mode : wgpu::PresentMode::Mailbox,

		};

		let swap = device.create_swap_chain(&surface, &sc_desc);

		Self {
			win_size,
			surface,
			adapter,
			device,
			queue,
			sc_desc,
			swap,	
		}

	}

	pub fn resize(&mut self, size : winit::dpi::PhysicalSize<u32>) {
		self.win_size = size;
		self.sc_desc.width = size.width;
		self.sc_desc.height = size.height;
		self.swap = self.device.create_swap_chain(&self.surface, &self.sc_desc);
	}

}