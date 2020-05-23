#pragma once

#include "vendor/imgui/imgui.h"
#include "vendor/imgui/imfilebrowser.h"
#include "vendor/imgui/imgui_impl_glfw.h"
#include "vendor/imgui/imgui_impl_opengl3.h"
#include "vendor/audiofile/AudioFile.h"

#include <map>
#include <math.h>
#include <valarray>
#include <complex>
#include "vendor/imgui/imgui_plot.h"
#include <random>

namespace my_gui
{
	bool p_open = true;

	ImGui::FileBrowser fileDialog;

	void setup_file_dialog()
	{
		fileDialog.SetTitle("Open a file");
		fileDialog.SetTypeFilters({ ".wav" });
	}

	inline void init(const char* glsl_version, GLFWwindow* window)
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		//ImGui::StyleColorsLight();
		

		// Setup Platform/Renderer bindings
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);


		setup_file_dialog();
	}

	inline void create_frame()
	{
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	double logNormalize(double value, double max, double expectedMax)
	{
		return log(1 + value) / log(1 + max) * expectedMax;
	}

	float zcr(int numSamples, float *samples, float srate, int start, float len) {
		int sum = 0;
		bool sign = false;

		for (int i = start; i < start + len; i++) {
			if ((!sign && samples[i] < 0) || (sign && samples[i] > 0)) {
				sum++;
				sign = !sign;
			}
		}
		return sum/((len/srate) * 2); // TODO: Change seconds to selection time
	}

	float autocorrelation(int numSamples, float* samples, int sampling, float step, int start, float len) { // TODO: Match the selection

		auto max_sample = *std::max_element(samples, samples + numSamples);
		for (int i = 0; i < numSamples; i++) {
			samples[i] *= 1 / max_sample;
		}

		std::vector<float> frequenciesFound;
		float sum;
		int windowIndex = 0;
		std::vector<std::vector<float>> windowSamples;
		int windowSize = (numSamples * step) / (numSamples / sampling);

		for (int i = 0; i < (1 / step) * (numSamples / sampling); i++) {
			std::vector<float> window;
			for (int s = 0; s < windowSize; s++) {
				window.push_back(samples[s + (windowSize * windowIndex)]);
			}
			windowSamples.push_back(window);
			windowIndex++;
		}

		std::vector<std::vector<float>> autocorrelationFunctions(windowSamples.size());
		windowIndex = 0;
		for (auto window : windowSamples) {
			for (int i = 0; i < window.size(); i++) {
				sum = 0;
				for (int j = 0; j < window.size() - i; j++) {
					sum += (window[j + i] * window[j]);
				}
				autocorrelationFunctions[windowIndex].push_back(sum / numSamples);
			}
			windowIndex++;
		}

		for (auto value : autocorrelationFunctions) {
			float prevValue = 0;
			float nextValue = 0;
			auto zeroElement = value[0];
			for (int i = 1; i < value.size() - 1; i++) {
				prevValue = value[i - 1];
				nextValue = value[i + 1];

				if (prevValue < value[i] && nextValue < value[i]) {
					auto freq = sampling / i;
					frequenciesFound.push_back(freq);
					break;
				}
			}
		}

		int median;
		float pitch = -1;

		std::sort(frequenciesFound.begin(), frequenciesFound.end());
		if (frequenciesFound.size() > 0) {
			if (frequenciesFound.size() % 2 == 0) {
				int middle = frequenciesFound.size() / 2;
				median = (frequenciesFound[middle] + frequenciesFound[middle + 1]) / 2;
			}
			else {
				median = frequenciesFound[frequenciesFound.size() / 2];
			}
			pitch = median;
		}


		return pitch;
	}

	typedef std::complex<double> ComplexVal;
	typedef std::valarray<ComplexVal> SampleArray;

	void fft(SampleArray& x)
	{
		// DFT
		unsigned int N = x.size(), k = N, n;
		double thetaT = 3.14159265358979323846264338328L / N;
		ComplexVal phiT = ComplexVal(cos(thetaT), -sin(thetaT)), T;
		while (k > 1)
		{
			n = k;
			k >>= 1;
			phiT = phiT * phiT;
			T = 1.0L;
			for (unsigned int l = 0; l < k; l++)
			{
				for (unsigned int a = l; a < N; a += n)
				{
					unsigned int b = a + k;
					ComplexVal t = x[a] - x[b];
					x[a] += x[b];
					x[b] = t * T;
				}
				T *= phiT;
			}
		}
		// Decimate
		unsigned int m = (unsigned int)log2(N);
		for (unsigned int a = 0; a < N; a++)
		{
			unsigned int b = a;
			// Reverse bits
			b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
			b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
			b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
			b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
			b = ((b >> 16) | (b << 16)) >> (32 - m);
			if (b > a)
			{
				ComplexVal t = x[a];
				x[a] = x[b];
				x[b] = t;
			}
		}
	}

	void preemphasis(SampleArray& values, double alpha) {
		auto size = values.size();
		for (int i = 1; i < size; i++) {
			values[i] -= alpha * values[i - 1];

		}
	}

	void hamming(SampleArray& values) {
		auto size = values.size();
		for (int i = 0; i < size; i++) {
			values[i] *= 0.53836 - (0.46164 * std::cos((2 * 3.14 * i) / (values.size() - 1.0)));
		}
	}

	void gauss(SampleArray& values) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(0.0, 4.5);
		auto size = values.size();
		for (int i = 0; i < size; i++) {

			values[i] *= std::exp(-0.5 * std::pow(((i - (size - 1.0) / 2.0) / (distribution(generator) * ((size - 1.0) / 2.0))), 2));
		}
	}

	void hanning(SampleArray& values) {
		auto size = values.size();
		for (int i = 0; i < size; i++) {
			values[i] *= 0.5 * (1 - std::cos((2 * 3.14 * i) / size - 1));
		}
	}

	void bartlett(SampleArray& values) {
		auto size = values.size();
		for (int i = 0; i < size; i++) {
			values[i] *= 2.0 / (size - 1.0) * ((size - 1.0) / 2.0 - std::abs(i - (size - 1.0) / 2.0));
		}
	}

	uint32_t sanitize(uint32_t v) {
		return std::pow(2, std::ceil(std::log(v) / std::log(2.0)));
	}

	inline void create_mainmenu_window()

	{
		ImGui::BeginMainMenuBar();
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open"))
			{
				fileDialog.Open();
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();


		static bool show_output = false;
		static bool count_autocor = false;
		static bool show_comb = false;
		static bool show_results = false;
		static bool show_options = false;

		static bool invoke_fft = false;

		static AudioFile<float> audioFile;
		static float pitch_zcr = -1;
		static float pitch_comb = -1;
		static std::string file_path;
		static uint32_t selection_start = 0, selection_length = 0;
		static SampleArray fft_vals;

		static int windowFunctionRadio = 0;

		fileDialog.Display();
		if (fileDialog.HasSelected())
		{
			std::cout << "Selected filename" << fileDialog.GetSelected().string() << std::endl;
			audioFile.load(fileDialog.GetSelected().string());
			file_path = fileDialog.GetSelected().string();
			show_output = true;
			
			fileDialog.ClearSelected();
		}
		if (count_autocor) {

			count_autocor = false;
		}

		if (show_output) {
			ImGui::Begin("Audio Plots", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		

			ImGui::PlotConfig conf;
			conf.values.ys = audioFile.samples[0].data();
			conf.values.count = audioFile.getNumSamplesPerChannel();
			conf.scale.min = -1.1;
			conf.scale.max = 1.1;
			conf.grid_y.show = true;
			conf.tooltip.show = true;
			conf.tooltip.format = "x=%.2f, y=%.2f";
			conf.grid_x.show = true;
			conf.grid_x.subticks = 1;
			conf.selection.show = true;
			conf.grid_y.show = true;
			conf.frame_size = ImVec2(1200, 250);
			conf.line_thickness = 0.2f;
			conf.values.color = ImColor(255, 20, 147);
			conf.selection.show = true;
			conf.selection.start = &selection_start;
			conf.selection.length = &selection_length;
			conf.selection.sanitize_fn = sanitize;
			//ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(255, 255 ,255));

			if ((ImGui::Plot("Audio", conf) == ImGui::PlotStatus::selection_updated && selection_length > 0) || (invoke_fft && selection_length > 0)) {
				pitch_zcr = zcr(audioFile.getNumSamplesPerChannel(), audioFile.samples[0].data(), audioFile.getSampleRate(), selection_start, selection_length);
				
				fft_vals.resize(selection_length, 0);

				for (int i = selection_start, index = 0; i < selection_start + selection_length; i++, index++) {
					fft_vals[index] = audioFile.samples[0][i];
				}
				preemphasis(fft_vals, 0.95);
				switch (windowFunctionRadio)
				{
				case 0:
					break;
				case 1:
					gauss(fft_vals);
					break;
				case 2:
					hamming(fft_vals);
					break;
				case 3:
					hanning(fft_vals);
					break;
				case 4:
					bartlett(fft_vals);
					break;
				default:
					break;
				}
				fft(fft_vals);

				show_options = true;
				show_comb = true;
				show_results = true;
				invoke_fft = false;

				//pitch_autocor = autocorrelation(audioFile.getNumSamplesPerChannel(), audioFile.samples[0].data(), audioFile.getSampleRate(), 0.05, selection_start, selection_length);
			}

			
			//ImGui::PopStyleColor();

			conf.values.ys_list = nullptr;
			conf.selection.show = false;
			// set new ones
			conf.values.ys = audioFile.samples[0].data();
			//conf.frame_size = ImVec2(600, 250);
			conf.values.offset = selection_start;
			conf.values.count = selection_length;
			conf.line_thickness = 0.3f;

			ImGui::Plot("Selection", conf);

			conf.values.offset = 0;
			
			conf.values.ys_list = nullptr;
			conf.selection.show = false;
			// set new ones
			std::vector<float> plot_fft;
			std::vector<float> fft_frequencies;
			std::vector<float> comb_sums;

			for (int i = 0; i < fft_vals.size()/2; i++) {
				plot_fft.push_back(std::abs(fft_vals[i]));
				fft_frequencies.push_back((audioFile.getSampleRate() / (fft_vals.size()/2)) * i);
			}

			
			conf.values.count = plot_fft.size();
			if (plot_fft.size() > 0) {
				auto it = *max_element(std::begin(plot_fft), std::end(plot_fft));
				conf.scale.max = it;

			}
			else {
				conf.scale.max = 0;
			}

			if (show_comb) {

				int best = 0, best_step = 0;
				for (int step = 3; step < 200; ++step) {
					int sum = 0;
					for (int i = 1; i < 20 && i * step < plot_fft.size(); ++i) {
						for (int di = 0; di < i; ++di) {
							sum += plot_fft[i * step + di] / i;
						}
					}
					
					sum *= 10000;
					comb_sums.push_back(sum);
				}

				int sum = 0;
				for (int i = 0; i < comb_sums.size(); ++i) {
					sum = comb_sums.at(i);
					if (sum > best) {
						best_step = i;
						best = sum;
					}
				}
				show_comb = false;
				pitch_comb = (best_step * audioFile.getSampleRate()) / plot_fft.size();
			}

			conf.scale.min = 0;		
			conf.grid_y.size = plot_fft.size();
			conf.values.ys = plot_fft.data();
			conf.grid_x.size = fft_frequencies.size();
			conf.values.xs = fft_frequencies.data();
			conf.values.color = ImColor(0, 255, 100);

			conf.line_thickness = 0.3f;
			ImGui::Plot("fft", conf);

			ImGui::End();
		}

		if (show_options) {
			ImGui::Begin("Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("Window functions");
			if (ImGui::RadioButton("Rectangular Window", &windowFunctionRadio, 0))
				invoke_fft = true;
			if(ImGui::RadioButton("Gauss Window", &windowFunctionRadio, 1))
				invoke_fft = true;
			if(ImGui::RadioButton("Hamming Window", &windowFunctionRadio, 2))
				invoke_fft = true;
			if(ImGui::RadioButton("Hanning Window", &windowFunctionRadio, 3))
				invoke_fft = true;
			if(ImGui::RadioButton("Bartlett Window", &windowFunctionRadio, 4))
				invoke_fft = true;
			ImGui::End();
		}

		if (show_results) {
			ImGui::Begin("Results", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("ZCR Frequency: ");
			ImGui::Text(std::to_string(pitch_zcr).c_str());
			ImGui::Text("Comb Filtration Frequency: ");
			ImGui::Text(std::to_string(pitch_comb).c_str());
			ImGui::End();
		}


	};


}
