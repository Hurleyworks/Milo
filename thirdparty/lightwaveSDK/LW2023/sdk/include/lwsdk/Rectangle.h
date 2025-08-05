// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_RECTANGLE_INCLUDED
#define LWCPP_RECTANGLE_INCLUDED

#include <limits>
#include <lwsdk/Vector.h>

namespace lwsdk
{
    template <class T>
    class Rectangle
    {
    public:
        typedef Vector<2, T> Point2;
        typedef Vector<3, T> Point3;

        // left, top, right, bottom
        T x1, y1, x2, y2;

        /// Creates a default rectangle.
        Rectangle();

        template <class T2>
        explicit Rectangle(const Rectangle<T2>& other);

        /// Creates a rectangle based on a position and a size.
        static Rectangle sized(T left, T top, T width, T height);

        Rectangle(T x1, T y1, T x2, T y2);
        Rectangle(const Point2& corner1, const Point2& corner2);
        Rectangle(const Point3& corner1, const Point3& corner2);

        Point2 top_left() const;
        Point2 top_right() const;
        Point2 bottom_left() const;
        Point2 bottom_right() const;
        Point2 center() const;

        T left() const;
        T right() const;
        T top() const;
        T bottom() const;
        T width() const;
        T height() const;
        Vector<2, T> size() const;

        /// @return True if the rectangle's size is less than or equal to zero.
        bool empty() const;

        void set_size(const Vector<2, T>& s);

        /// Preserves the size of the rectangle
        void move_top_left_to(const Vector<2, T>& s);
        void move_top_left_by(const Vector<2, T>& s);

        void expand(T x, T y);
        void expand(const Point2& other);
        void expand(const Point3& other);
        void expand(const Rectangle& other);
        void expand(T amount);
        void pad(T percentage);

        /// Crops this rectangle so that it fits inside the specified rectangle.
        void crop(const Rectangle<T>& rect);

        /// @return An adjusted version of the rectangle with the coordinates
        /// offset by the specified amounts.
        Rectangle<T> adjusted(T left_offset, T top_offset, T right_offset, T bottom_offset) const;

        /// @return true if the specified point is inside this rectangle.
        bool contains(T px, T py) const;

        /// @return true if the specified point is inside this rectangle.
        bool contains(const Point2& point) const;

        /// @return true if this rectangle is completely inside the other.
        bool inside(const Rectangle& other) const;

    private:
        static T choose_min(T a, T b);
        static T choose_max(T a, T b);
        void init(T i_x1, T i_y1, T i_x2, T i_y2);
    };

    template <class T>
    void subdivide(const Rectangle<T>& rect, Rectangle<T> out_sub_rects[4])
    {
        const T xmin = rect.left(), xmax = rect.right(), xcen = (xmax - xmin) / 2 + xmin;
        const T ymin = rect.top(), ymax = rect.bottom(), ycen = (ymax - ymin) / 2 + ymin;
        out_sub_rects[0] = Rectangle<T>(Vector<2, T>(xmin, ymin), Vector<2, T>(xcen, ycen));
        out_sub_rects[1] = Rectangle<T>(Vector<2, T>(xmin, ycen), Vector<2, T>(xcen, ymax));
        out_sub_rects[2] = Rectangle<T>(Vector<2, T>(xcen, ymin), Vector<2, T>(xmax, ycen));
        out_sub_rects[3] = Rectangle<T>(Vector<2, T>(xcen, ycen), Vector<2, T>(xmax, ymax));
    }

    /// Resizes the rectangle to have the new width and height while
    /// maintaining its top-left corner.
    template <class T>
    void resize(Rectangle<T>& rect, const Vector<2, T>& new_size)
    {
        rect.x2 = rect.x1 + new_size[0];
        rect.y2 = rect.y1 + new_size[1];
    }

    /// Moves the rectangle so that its top-left corner is at the
    /// specified position.
    template <class T>
    void move(Rectangle<T>& rect, const Vector<2, T>& new_top_left)
    {
        T x_offset = new_top_left[0] - rect.left();
        rect.x1 += x_offset;
        rect.x2 += x_offset;

        T y_offset = new_top_left[1] - rect.top();
        rect.y1 += y_offset;
        rect.y2 += y_offset;
    }

    /// @return a rectangle which has been resized from the top-left corner.
    template <class T>
    Rectangle<T> resized(const Rectangle<T>& rect, const Vector<2, T>& new_size)
    {
        Rectangle<T> new_rect(rect);
        resize(new_rect, new_size);
        return new_rect;
    }

    /// @return a rectangle which has been moved so that its top-left corner
    /// is at the specified position.
    template <class T>
    Rectangle<T> moved(const Rectangle<T>& rect, const Vector<2, T>& new_top_left)
    {
        Rectangle<T> new_rect(rect);
        move(new_rect, new_top_left);
        return new_rect;
    }

    template <class T>
    bool intersect(const Vector<2, T>& point, const Rectangle<T>& rect)
    {
        return point[0] >= rect.left()
            && point[0] <= rect.right()
            && point[1] >= rect.top()
            && point[1] <= rect.bottom();
    }

    template <class T>
    bool intersect(const Rectangle<T>& rect, const Vector<2, T>& point)
    {
        return intersect(point, rect);
    }

    template <class T>
    bool intersect(const Rectangle<T>& rect1, const Rectangle<T>& rect2)
    {
        return rect1.left() <= rect2.right()
            && rect1.right() >= rect2.left()
            && rect1.top() <= rect2.bottom()
            && rect1.bottom() >= rect2.top();
    }

    // ----------------------------------------------------------------------
    // implementation details
    // ----------------------------------------------------------------------
    template <class T>
    T Rectangle<T>::choose_min(T a, T b)
    {
        return a < b ? a: b;
    }

    template <class T>
    T Rectangle<T>::choose_max(T a, T b)
    {
        return a > b ? a: b;
    }

    template <class T>
    Rectangle<T> Rectangle<T>::sized(T left, T top, T width, T height)
    {
        return Rectangle<T>(left, top, left + width, top + height);
    }

    template <class T>
    Rectangle<T>::Rectangle()
    {
        T max_val = std::numeric_limits<T>::max();
        x1 = y1 = max_val;
        x2 = y2 = -max_val;
    }

    template <class T>
    template <class T2>
    Rectangle<T>::Rectangle(const Rectangle<T2>& other)
    {
        x1 = static_cast<T>(other.x1);
        x2 = static_cast<T>(other.x2);
        y1 = static_cast<T>(other.y1);
        y2 = static_cast<T>(other.y2);
    }

    template <class T>
    void Rectangle<T>::init(T i_x1, T i_y1, T i_x2, T i_y2)
    {
        x1 = choose_min(i_x1, i_x2);
        x2 = choose_max(i_x1, i_x2);
        y1 = choose_min(i_y1, i_y2);
        y2 = choose_max(i_y1, i_y2);
    }

    template <class T>
    Rectangle<T>::Rectangle(T i_x1, T i_y1, T i_x2, T i_y2)
    {
        init(i_x1, i_y1, i_x2, i_y2);
    }

    template <class T>
    Rectangle<T>::Rectangle(const Point2& corner1, const Point2& corner2)
    {
        init(corner1[0], corner1[1], corner2[0], corner2[1]);
    }

    template <class T>
    Rectangle<T>::Rectangle(const Point3& corner1, const Point3& corner2)
    {
        init(corner1[0], corner1[1], corner2[0], corner2[1]);
    }

    template <class T>
    T Rectangle<T>::left() const
    {
        return x1;
    }

    template <class T>
    T Rectangle<T>::right() const
    {
        return x2;
    }

    template <class T>
    T Rectangle<T>::top() const
    {
        return y1;
    }

    template <class T>
    T Rectangle<T>::bottom() const
    {
        return y2;
    }

    template <class T>
    T Rectangle<T>::width() const
    {
        return right() - left();
    }

    template <class T>
    T Rectangle<T>::height() const
    {
        return bottom() - top();
    }

    template <class T>
    Vector<2, T> Rectangle<T>::size() const
    {
        return Vector<2, T>(width(), height());
    }

    template <class T>
    bool Rectangle<T>::empty() const
    {
        return right() <= left() || bottom() <= top();
    }

    template <class T>
    void Rectangle<T>::set_size(const Vector<2, T>& s)
    {
        x2 = x1 + s[0];
        y2 = y1 + s[1];
    }

    template <class T>
    void Rectangle<T>::move_top_left_to(const Vector<2, T>& s)
    {
        x2 += s[0] - x1;
        y2 += s[1] - y1;
        x1 = s[0];
        y1 = s[1];
    }

    template <class T>
    void Rectangle<T>::move_top_left_by(const Vector<2, T>& s)
    {
        x2 += s[0];
        y2 += s[1];
        x1 += s[0];
        y1 += s[1];
    }


    template <class T>
    typename Rectangle<T>::Point2 Rectangle<T>::top_left() const
    {
        return Point2(left(), top());
    }

    template <class T>
    typename Rectangle<T>::Point2 Rectangle<T>::top_right() const
    {
        return Point2(right(), top());
    }

    template <class T>
    typename Rectangle<T>::Point2 Rectangle<T>::bottom_left() const
    {
        return Point2(left(), bottom());
    }

    template <class T>
    typename Rectangle<T>::Point2 Rectangle<T>::bottom_right() const
    {
        return Point2(right(), bottom());
    }

    template <class T>
    typename Rectangle<T>::Point2 Rectangle<T>::center() const
    {
        Point2 cen = (bottom_right() - top_left()) / 2;
        return cen + top_left();
    }

    template <class T>
    void Rectangle<T>::pad(T percentage)
    {
        T w = width(), h = height();
        x1 -= w * percentage;
        x2 += w * percentage;
        y1 -= h * percentage;
        y2 += h * percentage;
    }

    template <class T>
    void Rectangle<T>::expand(T x, T y)
    {
        x1 = choose_min(x1, x);
        x2 = choose_max(x2, x);
        y1 = choose_min(y1, y);
        y2 = choose_max(y2, y);
    }

    template <class T>
    void Rectangle<T>::expand(const Point2& other)
    {
        x1 = choose_min(x1, other[0]);
        x2 = choose_max(x2, other[0]);
        y1 = choose_min(y1, other[1]);
        y2 = choose_max(y2, other[1]);
    }

    template <class T>
    void Rectangle<T>::expand(const Point3& other)
    {
        expand(Point2(other));
    }

    template <class T>
    void Rectangle<T>::expand(const Rectangle& other)
    {
        x1 = choose_min(x1, other.x1);
        x2 = choose_max(x2, other.x2);
        y1 = choose_min(y1, other.y1);
        y2 = choose_max(y2, other.y2);
    }

    template <class T>
    void Rectangle<T>::expand(T amount)
    {
        x1 -= amount;
        x2 += amount;
        y1 -= amount;
        y2 += amount;
    }

    template <class T>
    Rectangle<T> Rectangle<T>::adjusted(T left_offset, T top_offset, T right_offset, T bottom_offset) const
    {
        return Rectangle<T>(x1 + left_offset, y1 + top_offset,
                            x2 + right_offset, y2 + bottom_offset);
    }

    template <class T>
    bool Rectangle<T>::contains(T px, T py) const
    {
        return px >= left() && px <= right() &&
               py >= top() && py <= bottom();
    }

    template <class T>
    bool Rectangle<T>::contains(const Point2& point) const
    {
        return contains(point[0], point[1]);
    }

    template <class T>
    bool Rectangle<T>::inside(const Rectangle& other) const
    {
        return left() >= other.left()
            && top() >= other.top()
            && right() <= other.right()
            && bottom() <= other.bottom();
    }

    template <class T>
    void Rectangle<T>::crop(const Rectangle<T>& rect)
    {
        if (left() < rect.left())
            x1 = rect.left();
        if (right() > rect.right())
            x2 = rect.right();
        if (top() < rect.top())
            y1 = rect.top();
        if (bottom() > rect.bottom())
            y2 = rect.bottom();
    }

    template <class T>
    bool operator==(const Rectangle<T>& lhs, const Rectangle<T>& rhs)
    {
        return lhs.left() == rhs.left() &&
               lhs.top() == rhs.top() &&
               lhs.bottom() == rhs.bottom() &&
               lhs.right() == rhs.right();
    }

    template <class T>
    bool operator!=(const Rectangle<T>& lhs, const Rectangle<T>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T>
    bool operator<(const Rectangle<T>& lhs, const Rectangle<T>& rhs)
    {
        const T lhs_values[4] = {lhs.x1, lhs.y1, lhs.x2, lhs.y2};
        const T rhs_values[4] = {rhs.x1, rhs.y1, rhs.x2, rhs.y2};
        for (int j=0; j < 4; ++j)
        {
            if (lhs_values[j] < rhs_values[j])
                return true;
            else if (lhs_values[j] > rhs_values[j])
                return false;
        }
        return false;
    }

    template <class T>
    bool operator>(const Rectangle<T>& lhs, const Rectangle<T>& rhs)
    {
        return rhs < lhs;
    }
}

#endif