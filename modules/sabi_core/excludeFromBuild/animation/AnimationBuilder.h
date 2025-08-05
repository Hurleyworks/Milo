

class AnimationBuilder
{
 public:
    using EasingFunction = std::function<float (float)>;

    AnimationBuilder (const std::string& targetNode) :
        targetNode_ (targetNode) {}

    // Add a keyframe with multiple properties
    AnimationBuilder& addKeyframe (float time,
                                   const Eigen::Vector3f& translation = Eigen::Vector3f::Zero(),
                                   const Eigen::Quaternionf& rotation = Eigen::Quaternionf::Identity(),
                                   const Eigen::Vector3f& scale = Eigen::Vector3f::Ones())
    {
        KeyFrame keyframe;
        keyframe.time = time;
        keyframe.translation = translation;
        keyframe.rotation = rotation;
        keyframe.scale = scale;
        keyframes_.push_back (keyframe);
        return *this;
    }

    // Set duration
    AnimationBuilder& setDuration (float duration)
    {
        duration_ = duration;
        return *this;
    }

    // Set easing function
    AnimationBuilder& setEasing (EasingFunction easingFunc)
    {
        easingFunc_ = easingFunc;
        return *this;
    }

    // Set looping
    AnimationBuilder& setLooping (bool loop)
    {
        loop_ = loop;
        return *this;
    }

    // Build the AnimationChannels
    std::vector<AnimationChannel> build()
    {
        std::vector<AnimationChannel> channels;

        if (keyframes_.empty())
        {
            LOG (WARNING) << "No keyframes defined for animation on node: " << targetNode_;
            return channels;
        }

        std::map<fastgltf::AnimationPath, AnimationChannel> channelMap;

        // Initialize channels for each path
        channelMap[fastgltf::AnimationPath::Translation].path = fastgltf::AnimationPath::Translation;
        channelMap[fastgltf::AnimationPath::Rotation].path = fastgltf::AnimationPath::Rotation;
        channelMap[fastgltf::AnimationPath::Scale].path = fastgltf::AnimationPath::Scale;

        float lastKeyframeTime = 0.0f;

        for (auto& keyframe : keyframes_)
        {
            keyframe.time *= duration_;
            lastKeyframeTime = std::max (lastKeyframeTime, keyframe.time);

            // Apply easing if set
            if (easingFunc_)
            {
                keyframe.time = easingFunc_ (keyframe.time / duration_) * duration_;
            }

            channelMap[fastgltf::AnimationPath::Translation].keyFrames.push_back (keyframe);
            channelMap[fastgltf::AnimationPath::Rotation].keyFrames.push_back (keyframe);
            channelMap[fastgltf::AnimationPath::Scale].keyFrames.push_back (keyframe);
        }

        // Add looping keyframe if needed
        if (loop_ && !keyframes_.empty())
        {
            KeyFrame loopKeyframe = keyframes_.front();
            loopKeyframe.time = lastKeyframeTime + 0.001f; // Slightly after the last keyframe

            channelMap[fastgltf::AnimationPath::Translation].keyFrames.push_back (loopKeyframe);
            channelMap[fastgltf::AnimationPath::Rotation].keyFrames.push_back (loopKeyframe);
            channelMap[fastgltf::AnimationPath::Scale].keyFrames.push_back (loopKeyframe);
        }

        // Finalize channels
        for (auto& [path, channel] : channelMap)
        {
            channel.targetNode = targetNode_;
            channels.push_back (channel);
        }

        return channels;
    }

 private:
    std::string targetNode_;
    std::vector<KeyFrame> keyframes_;
    float duration_ = 5.0f;
    EasingFunction easingFunc_ = nullptr;
    bool loop_ = true;
};

// Helper function for linear interpolation (default easing)
inline float linearEase (float t)
{
    return t;
}