#pragma once

class GltfAnimationExporter
{
 public:
    std::vector<AnimationChannel> exportAnimation (const fastgltf::Asset& asset, size_t animationIndex)
    {
        std::vector<AnimationChannel> channels;

        if (animationIndex >= asset.animations.size())
        {
            LOG (CRITICAL) << "Invalid animation index: " << animationIndex;
            return channels;
        }

        const auto& animation = asset.animations[animationIndex];

        for (const auto& channel : animation.channels)
        {
            AnimationChannel animChannel;
            animChannel.targetNode = asset.nodes[*channel.nodeIndex].name;
            animChannel.path = channel.path;

            const auto& sampler = animation.samplers[channel.samplerIndex];
            const auto& input = asset.accessors[sampler.inputAccessor];
            const auto& output = asset.accessors[sampler.outputAccessor];

            auto times = getAccessorData<float> (asset, input);

            switch (channel.path)
            {
                case fastgltf::AnimationPath::Translation:
                case fastgltf::AnimationPath::Scale:
                case fastgltf::AnimationPath::Rotation:
                {
                    auto values = getAccessorData<float> (asset, output);

                    auto componentCount = fastgltf::getNumComponents (output.type);
                    
                    for (size_t i = 0; i < times.size(); ++i)
                    {
                        KeyFrame keyFrame = createKeyFrame (channel, times[i], &values[i * componentCount]);

                        keyFrame.debug();
                        animChannel.keyFrames.push_back (keyFrame);
                    }

                    break;
                }

                default:
                    LOG (CRITICAL) << "Unsupported animation path";
                    continue;
            }

            channels.push_back (std::move (animChannel));
        }

        return channels;
    }

 private:
    template <typename T>
    std::vector<T> getAccessorData (const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
    {
        auto componentCount = fastgltf::getNumComponents (accessor.type);
        std::vector<T> data (accessor.count * componentCount);

        fastgltf::copyComponentsFromAccessor<T> (asset, accessor, data.data());

        return data;
    }

    KeyFrame createKeyFrame (const auto& channel, float time, const float* values)
    {
        KeyFrame keyFrame;
        keyFrame.time = time;

        switch (channel.path)
        {
            case fastgltf::AnimationPath::Translation:
                keyFrame.translation = Eigen::Vector3f (values[0], values[1], values[2]);
                keyFrame.rotation = Eigen::Quaternionf::Identity(); // Set identity rotation
                keyFrame.scale = Eigen::Vector3f::Ones();           // Set unit scale
                break;
            case fastgltf::AnimationPath::Rotation:
                keyFrame.translation = Eigen::Vector3f::Zero(); // Set zero translation
                keyFrame.rotation = Eigen::Quaternionf (values[3], values[0], values[1], values[2]).normalized();
                keyFrame.scale = Eigen::Vector3f::Ones(); // Set unit scale
                break;
            case fastgltf::AnimationPath::Scale:
                keyFrame.translation = Eigen::Vector3f::Zero();     // Set zero translation
                keyFrame.rotation = Eigen::Quaternionf::Identity(); // Set identity rotation
                keyFrame.scale = Eigen::Vector3f (values[0], values[1], values[2]);
                break;
            default:
                LOG (CRITICAL) << "Unsupported animation path in createKeyFrame";
                // Set default values for all fields
                keyFrame.translation = Eigen::Vector3f::Zero();
                keyFrame.rotation = Eigen::Quaternionf::Identity();
                keyFrame.scale = Eigen::Vector3f::Ones();
        }

        return keyFrame;
    }
};