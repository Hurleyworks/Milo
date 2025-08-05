

using namespace Eigen;

CameraBody::CameraBody()
{
    pose.setIdentity();
}

CameraBody::~CameraBody()
{
}

void CameraBody::lookAt (const Eigen::Vector3f& eyePoint, const Eigen::Vector3f& target, const Eigen::Vector3f& up)
{
    pose.translation() = eyePoint;
    this->target = target;
    Eigen::Vector3f f = (target - eyePoint).normalized();

    eye = eyePoint;
    viewDirection = f;

   // mace::vecStr3f (f, DBUG, "View direction");

    Matrix3f camAxes;
    camAxes.col (2) = -f;
    camAxes.col (0) = worldUp.cross (camAxes.col (2)).normalized();
    camAxes.col (1) = camAxes.col (2).cross (camAxes.col (0)).normalized();
    orientation = Quaternionf (camAxes);

    forward = -(orientation * Vector3f::UnitZ());

    pose.linear() = -orientation.toRotationMatrix();

    calcViewMatrix(); // Call this to update mU and mV
    viewMatrixCached = true;
}

void CameraBody::lookAt (const Eigen::Vector3f& eyePoint, const Eigen::Vector3f& target)
{
   
    this->target = target;
    eye = eyePoint;

    pose.translation() = eye;

    Eigen::Vector3f f = (target - eyePoint).normalized();
    viewDirection = f;

    /// Check for degeneracies.If the upDir and targetDir are parallel
    // or opposite, then compute a new, arbitrary up direction that is
    // not parallel or opposite to the targetDir.
    Vector3f upDir = worldUp;

    if (upDir.cross (f).squaredNorm() == 0)
    {
        upDir = f.cross (Vector3f::UnitX());
        if (upDir.squaredNorm() == 0)
            upDir = f.cross (Vector3f::UnitZ());

        upDir *= -1.0f; // to match Cinder
    }

    Matrix3f camAxes;
    camAxes.col (2) = -f;
    camAxes.col (0) = upDir.cross (camAxes.col (2)).normalized();
    camAxes.col (1) = camAxes.col (2).cross (camAxes.col (0)).normalized();
    orientation = Quaternionf (camAxes);

    forward = -(orientation * Vector3f::UnitZ());

    pose.linear() = orientation.toRotationMatrix();

    calcViewMatrix(); // Call this to update mU and mV
    viewMatrixCached = true;
}

void CameraBody::lookAt (const Eigen::Vector3f& eyePoint)
{
    eye = eyePoint;

    pose.translation() = eye;

    Eigen::Vector3f f = (target - eyePoint).normalized();
    viewDirection = f;

    /// Check for degeneracies.If the upDir and targetDir are parallel
    // or opposite, then compute a new, arbitrary up direction that is
    // not parallel or opposite to the targetDir.
    Vector3f upDir = worldUp;

    if (upDir.cross (f).squaredNorm() == 0)
    {
        upDir = f.cross (Vector3f::UnitX());
        if (upDir.squaredNorm() == 0)
            upDir = f.cross (Vector3f::UnitZ());

        upDir *= -1.0f; // to match Cinder
    }

    Matrix3f camAxes;
    camAxes.col (2) = -f;
    camAxes.col (0) = upDir.cross (camAxes.col (2)).normalized();
    camAxes.col (1) = camAxes.col (2).cross (camAxes.col (0)).normalized();
    orientation = Quaternionf (camAxes);

    forward = -(orientation * Vector3f::UnitZ());

    pose.linear() = orientation.toRotationMatrix();

    calcViewMatrix(); // Call this to update mU and mV
    viewMatrixCached = true;
}

void CameraBody::rotateAroundTarget (const Eigen::Quaternionf& q)
{
    // update the transform matrix
    if (!viewMatrixCached)
        calcViewMatrix();

    Vector3f t = viewMatrix * target;

    viewMatrix = Translation3f (t) * q * Translation3f (-t) * viewMatrix;

    Quaternionf qa (viewMatrix.linear());
    qa = qa.conjugate();
    orientation = qa;

    eye = -(qa * viewMatrix.translation());

    pose.translation() = eye;

    forward = -(orientation * Vector3f::UnitZ());
    viewDirection = (target - eye).normalized();

    pose.linear() = orientation.toRotationMatrix();

    viewMatrixCached = false;
}

void CameraBody::zoom (float d)
{
    float dist = (eye - target).norm();
    if (dist > d)
    {
        eye = eye + viewDirection * d;
        forward = -(orientation * Vector3f::UnitZ());
        viewMatrixCached = false;
        pose.translation() = eye;
    }
}

void CameraBody::track (const Eigen::Vector2f& point2D)
{
    Vector3f newPoint3D;
    bool newPointOk = mapToSphere (point2D, newPoint3D);

    if (lastPointOk && newPointOk)
    {
        Vector3f axis = lastPoint3D.cross (newPoint3D).normalized();
        float cos_angle = lastPoint3D.dot (newPoint3D);
        if (std::abs (cos_angle) < 1.0)
        {
            float angle = 2.0f * acos (cos_angle);
            //if (mMode == Around)
            rotateAroundTarget (Quaternionf (AngleAxisf (angle, axis)));
            //else
            //mpCamera->localRotate (Quaternionf (AngleAxisf (-angle, axis)));
        }
    }

    lastPoint3D = newPoint3D;
    lastPointOk = newPointOk;
}

void CameraBody::calcMatrices() const
{
    if (!viewMatrixCached) calcViewMatrix();
}

void CameraBody::calcViewMatrix() const
{
    mW = viewDirection.normalized();
    mU = orientation * Vector3f::UnitX();
    mV = orientation * Vector3f::UnitY();

    Quaternionf q = orientation.conjugate();
    viewMatrix.linear() = q.toRotationMatrix();

    if (!wabi::isOrthogonal<float> (viewMatrix.linear()))
    {
        Matrix3f m = viewMatrix.linear();
        if (!wabi::reOrthogonalize (m))
        {
            throw std::runtime_error ("Could not fix non-orthogongal matrix");
        }

        viewMatrix.linear() = m;
    }

    viewMatrix.translation() = -(viewMatrix.linear() * eye);

    forward = -(orientation * Vector3f::UnitZ());

    viewMatrixCached = true;
    inverseModelViewCached = false;
}

void CameraBody::calcInverseView() const
{
}

bool CameraBody::mapToSphere (const Eigen::Vector2f& p2, Eigen::Vector3f& v3)
{
    int w = sensor.getPixelResolution().x();
    int h = sensor.getPixelResolution().y();

    if ((p2.x() >= 0) && (p2.x() <= w &&
                          (p2.y() >= 0) && (p2.y() <= h)))
    {
        double x = (double)(p2.x() - 0.5 * w) / (double)w;
        double y = (double)(0.5 * h - p2.y()) / (double)h;
        double sinx = sin (M_PI * x * 0.5);
        double siny = sin (M_PI * y * 0.5);
        double sinx2siny2 = sinx * sinx + siny * siny;

        v3.x() = sinx;
        v3.y() = siny;
        v3.z() = sinx2siny2 < 1.0 ? sqrt (1.0 - sinx2siny2) : 0.0;

        return true;
    }
    else
        return false;
}


 void CameraBody::panHorizontal (float distance)
{
    eye += distance * mU;
    target += distance * mU;
    viewMatrixCached = false; // Since the view matrix will change
}

void CameraBody::panVertical (float distance)
{
    eye += distance * mV;
    target += distance * mV;
    viewMatrixCached = false; // Since the view matrix will change
}

void CameraBody::debugLog() const
{
    LOG (DBUG) << "Camera Debug Information:";
    LOG (DBUG) << "------------------------";
    LOG (DBUG) << "Camera Name: " << name;
    LOG (DBUG) << "Eye Position: " << eye.transpose();
    LOG (DBUG) << "Target Position: " << target.transpose();
    LOG (DBUG) << "World Up: " << worldUp.transpose();
    LOG (DBUG) << "View Direction: " << viewDirection.transpose();
    LOG (DBUG) << "Forward Vector: " << forward.transpose();
    LOG (DBUG) << "Right Vector (U): " << mU.transpose();
    LOG (DBUG) << "Up Vector (V): " << mV.transpose();
    LOG (DBUG) << "Focal Length: " << focalLength << " m";
    LOG (DBUG) << "Aperture: " << aperture;
    LOG (DBUG) << "Vertical FOV: " << verticalFOVradians * 180.0f / M_PI << " degrees";
    LOG (DBUG) << "Pose Translation: " << pose.translation().transpose();
    LOG (DBUG) << "Pose Rotation: " << Eigen::Quaternionf (pose.rotation()).coeffs().transpose();

    // Log sensor information
    const CameraSensor* sensor = &this->sensor;
    Eigen::Vector2i resolution = sensor->getPixelResolution();
    LOG (DBUG) << "Sensor Information:";
    LOG (DBUG) << "  Resolution: " << resolution.x() << "x" << resolution.y();
    LOG (DBUG) << "  Aspect Ratio: " << sensor->getPixelAspectRatio();
    LOG (DBUG) << "  Sensor Size: " << sensor->getSensorSize().transpose() << " m";
    LOG (DBUG) << "  Pixel Size: " << sensor->pixelSize().transpose();

    // Log matrices
    LOG (DBUG) << "View Matrix:";
    LOG (DBUG) << viewMatrix.matrix();

    // Log state flags
    LOG (DBUG) << "Camera State:";
    LOG (DBUG) << "  Dirty: " << (dirty ? "Yes" : "No");
    LOG (DBUG) << "  View Matrix Cached: " << (viewMatrixCached ? "Yes" : "No");
    LOG (DBUG) << "  Inverse Model View Cached: " << (inverseModelViewCached ? "Yes" : "No");

    // Log changes
    LOG (DBUG) << "Changes since last update:";
    LOG (DBUG) << "  Position changed: " << ((eye != lastEye) ? "Yes" : "No");
    LOG (DBUG) << "  Target changed: " << ((target != lastTarget) ? "Yes" : "No");
    LOG (DBUG) << "  World Up changed: " << ((worldUp != lastWorldUp) ? "Yes" : "No");
    LOG (DBUG) << "  Focal Length changed: " << ((focalLength != lastFocalLength) ? "Yes" : "No");
    LOG (DBUG) << "  Aperture changed: " << ((aperture != lastAperture) ? "Yes" : "No");
    LOG (DBUG) << "  Pose changed: " << ((pose.matrix() != lastPose.matrix()) ? "Yes" : "No");
    LOG (DBUG) << "  Vertical FOV changed: " << ((verticalFOVradians != lastVerticalFOVradians) ? "Yes" : "No");

    LOG (DBUG) << "------------------------";
}