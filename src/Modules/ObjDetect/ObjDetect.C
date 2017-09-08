// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>

#include <jevois/Debug/Log.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>

#include <linux/videodev2.h> // for v4l2 pixel types

// See if this is actually needed
#include <jevoisbase/Components/Utilities/BufferedVideoReader.H>


#include <stdarg.h> // needed by tiny_dnn

// Defines used to optimize tiny-dnn.
#define CNN_USE_TBB
#undef CNN_USE_DOUBLE

#define DNN_USE_IMAGE_API 1

// All defines used by tiny-dnn, double check in the end
#include <fstream>
#include <iostream>

#include <tiny-dnn/tiny_dnn/tiny_dnn.h>
#include <tiny-dnn/tiny_dnn/nodes.h>
#include <tiny-dnn/tiny_dnn/config.h> // for float_t, etc. this does not include much code
#include <tiny-dnn/tiny_dnn/util/aligned_allocator.h> // for aligned_allocator
#include <tiny-dnn/tiny_dnn/util/util.h> // for index3d
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // used for debugging only, see imshow below

//Defines for the input resolution
#define RESOLUTION_WIDTH 640
#define RESOLUTION_HEIGHT 480

/*! This module detects object usign the cifar10 trained network

@author Bilal Parvez

TODO: Decide upon a video mapping
@videomapping1 YUYV 640 480 7.5 YUYV 640 480 7.5 Bilal ObjDetect
@email bilalp@kth.se
@address Landsberger Str. 425 , 81241 München
@copyright Copyright (C) 2017 by Bilal Parvez
@mainurl https://bparvez.github.io/
@license GPL v3
@distribution Unrestricted
@restrictions None */
class ObjDetect : public jevois::StdModule
{
	public:
		tiny_dnn::network<tiny_dnn::sequential> nn;

		/*
		 * @brief rescale output to 0-100
		 *
		 * @note TODO: Look into what this does, and why it does it
		 */
		template <typename Activation> double rescale(double x) {
			Activation a(1);
			return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
		}


		//! Constructor
		ObjDetect(std::string const & instance) : jevois::StdModule(instance), itsScoresStr(" ") {

		}

		//! Post Initialization loading of weights
		virtual void postInit() override
		{
			// My self trained weights
			const std::string wpath = absolutePath("tiny-dnn/CIFAR/cifar-weights");

			using conv    = tiny_dnn::convolutional_layer;
			using pool    = tiny_dnn::max_pooling_layer;
			using fc      = tiny_dnn::fully_connected_layer;
			using relu    = tiny_dnn::relu_layer;
			using softmax = tiny_dnn::softmax_layer;

			const size_t n_fmaps  = 32;  ///< number of feature maps for upper layer
			const size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
			const size_t n_fc = 64;  ///< number of hidden units in fully-connected layer

			nn << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same)  // C1
				<< pool(32, 32, n_fmaps, 2)                              // P2
				<< relu(16, 16, n_fmaps)                                 // activation
				<< conv(16, 16, 5, n_fmaps, n_fmaps, tiny_dnn::padding::same)  // C3
				<< pool(16, 16, n_fmaps, 2)                                    // P4
				<< relu(8, 8, n_fmaps)                                        // activation
				<< conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_dnn::padding::same)  // C5
				<< pool(8, 8, n_fmaps2, 2)                                    // P6
				<< relu(4, 4, n_fmaps2)                                       // activation
				<< fc(4 * 4 * n_fmaps2, n_fc)                                 // FC7
				<< fc(n_fc, 10) << softmax(10);                               // FC10

			try
			{
				std::ifstream ifs(wpath);
				ifs >> nn;
				LINFO("Loaded pre-trained weights from " << wpath);
			}
			catch (...)
			{
				LFATAL("Could not load pre-trained weights from " << wpath);
			}
		}

		//! Virtual destructor for safe inheritance
		virtual ~ObjDetect() { }

		//! Processing function
		virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
		{

			//TODO: Hardcoded should read the batches.meta.txt file.
			const std::array<const std::string, 10> names = {
				"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
				"ship", "truck",
			};

			static jevois::Timer itsProcessingTimer("Processing");
			static std::string itsLastObjectCateg;

			// Wait for next available camera image:
			jevois::RawImage inimg = inframe.get(true);

			// We only support YUYV pixels in this example, any resolution:
			inimg.require("input", 640, 480, V4L2_PIX_FMT_YUYV);

			itsProcessingTimer.start();

			// Wait for an image from our gadget driver into which we will put our results:
			jevois::RawImage outimg = outframe.get();
			outimg.require("output", 640, 480, V4L2_PIX_FMT_YUYV);
			switch (outimg.height) {
				case 480: break; //normal mode
				case 360: //Something is wrong
				default: LFATAL("Incorrent output height should be 480");
			}

			// Paste the original image to the top-left corner of the display:
			jevois::rawimage::paste(inimg, outimg, 0, 0);

			// Get the input size and shape of the network
			auto inshape = nn[0]->in_shape()[0];
			auto sz = inshape.size();

			// Converting the raw image to opencv mat format
			cv::Mat rawcvimage = jevois::rawimage::cvImage(inimg);

			// In imshow is the conversion wrong , or does imshow
			// show the default conversion of opencv what is bgr.
			// TODO: Find out the correct resolution
			cv::cvtColor(rawcvimage, rawcvimage, CV_YUV2RGB_YUYV);

			//Add debug image print
			cv::imshow("Input image to network", rawcvimage);cv::waitKey(1);

			// Resize according to the input sizes of the first layer of the ML network
			cv::resize(rawcvimage, rawcvimage, cv::Size(inshape.width_, inshape.height_), 0, 0, cv::INTER_AREA);


			// Convert input image to vec_t with values in [-1..1]:
			// If does not work change back swap 2.0F and *in++
			tiny_dnn::vec_t data(sz);
			unsigned char const * in = rawcvimage.data; tiny_dnn::float_t * out = &data[0];
			for (size_t i = 0; i < sz; ++i) *out++ = (*in++) * (2.0F / 255.0F) - 1.0F;

			// Launch Image classification
			auto res = nn.predict(data);
			std::vector<std::pair<double, int>> scores;

			// sort & print top-3
			for (int i = 0; i < 10; i++)
				scores.emplace_back(rescale<tiny_dnn::tanh_layer>(res[i]), i);

			sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

			// Create a string to show all scores:
			std::ostringstream oss;
			for (int i = 0; i < 3; i++)
				oss << names[scores[i].second] << "," << scores[i].first << std::endl;
			itsScoresStr = oss.str();

			// Check whether the highest score is very high and significantly higher than the second best:
			float best1 = scores[0].first, best2 = scores[1].first;

			// Update our display upon each "clean" recognition:
			if (best1 > 90.0F && best2 < 60.0F)
			{
				itsLastObjectCateg = names[scores[0].second];
				LINFO("Object recognition: best: " << itsLastObjectCateg <<" (" << best1 <<
						"), second best: " << names[scores[1].second] << " (" << best2 << ")");
			}

			// Handle the bottom of the banner for better readibility
			jevois::rawimage::drawFilledRect(outimg, 0, 410, outimg.width, outimg.height - 250, 0x8000);

			//One time define for the text color
			unsigned short const txtcol = jevois::yuyv::White;

			// Let camera know we are done processing the input image:
			inframe.done(); // NOTE: optional here, inframe destructor would call it anyway

			// Print all object scores:
			jevois::rawimage::writeText(outimg, itsScoresStr, 3, 480 - 40, txtcol);

			// Write any positively recognized object category:
			jevois::rawimage::writeText(outimg, itsLastObjectCateg.c_str(), 350, 480 - 40, txtcol);

			// Show processing fps:
			std::string const & fpscpu = itsProcessingTimer.stop();
			jevois::rawimage::writeText(outimg, fpscpu, 3, 480 - 13, txtcol);

			// Send the output image with our processing results to the host over USB:
			outframe.send(); // NOTE: optional here, outframe destructor would call it anyway
		}

	protected:
		std::string itsScoresStr;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ObjDetect);

