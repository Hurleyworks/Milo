


class RadialFlower : public LoadStrategy
{
 public:
    RadialFlower (int numPetals, float radius) :
        numPetals (numPetals), radius (radius), currentIndex (0)
    {
        setupPattern();
    }

    void addNextItem (SpaceTime& spacetime) override
    {
        if (currentIndex < positions.size())
        {
            spacetime.worldTransform.translation() = positions[currentIndex];
            currentIndex++;
        }
    }

    void reset() override
    {
        currentIndex = 0;
    }

    void incrementCount() override
    {
        // This method is not used in this strategy
    }

 private:
    int numPetals;
    float radius;
    int currentIndex;
    std::vector<Eigen::Vector3f> positions;

    void setupPattern()
    {
        // Center position
        positions.push_back (Eigen::Vector3f::Zero());

        // Outer petals
        for (int i = 0; i < numPetals; ++i)
        {
            float angle = 2 * M_PI * i / numPetals;
            float x = radius * std::cos (angle);
            float y = radius * std::sin (angle);
            positions.push_back (Eigen::Vector3f (x, y, 0));
        }

        // Inner petals
        float innerRadius = radius * 0.6f;
        for (int i = 0; i < numPetals; ++i)
        {
            float angle = 2 * M_PI * (i + 0.5f) / numPetals;
            float x = innerRadius * std::cos (angle);
            float y = innerRadius * std::sin (angle);
            positions.push_back (Eigen::Vector3f (x, y, 0));
        }

        // Smaller elements
        float smallRadius = radius * 0.3f;
        for (int i = 0; i < numPetals * 2; ++i)
        {
            float angle = 2 * M_PI * i / (numPetals * 2);
            float x = smallRadius * std::cos (angle);
            float y = smallRadius * std::sin (angle);
            positions.push_back (Eigen::Vector3f (x, y, 0));
        }
    }
};