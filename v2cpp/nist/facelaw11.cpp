// FACE LAW v2 behind NIST's FRTE 1:1 API (frvt11.h, API 6.0; frvt_structs.h 3.1).  Single-threaded, fork-safe, no
// files read or written, nothing printed: the law is compiled in.  Every call is stateless and wrapped so that a fatal
// error still returns a code (General Evaluation Specifications 7.2, 7.4).  Templates: tmpl.cpp, the full-light law
// (21,176 bytes; 8 bytes when creation failed).  Similarity: the symmetric score + 127 (the score is a mean of table
// entries clipped to +-127, so it is never negative); -1 with VerifTemplateError when either template failed.
#include <cmath>
#include <exception>
#include <memory>
#include <new>
#include "frvt11.h"
#include "../facelaw2.h"
#include "../tmpl.h"

namespace {

class FaceLaw11 : public FRVT_11::Interface {
public:
    FRVT::ReturnStatus initialize(const std::string &) override { return FRVT::ReturnStatus(FRVT::ReturnCode::Success); }

    FRVT::ReturnStatus createFaceTemplate(const std::vector<FRVT::Image> &faces, FRVT::TemplateRole role, std::vector<uint8_t> &templ,
                                          std::vector<FRVT::EyePair> &eyeCoordinates) override {
        try { return one_person(faces, role, templ, eyeCoordinates); }
        catch (const std::bad_alloc &) { failed(templ, eyeCoordinates, faces.size()); return FRVT::ReturnStatus(FRVT::ReturnCode::MemoryError); }
        catch (...) { failed(templ, eyeCoordinates, faces.size()); return FRVT::ReturnStatus(FRVT::ReturnCode::UnknownError); }
    }

    FRVT::ReturnStatus one_person(const std::vector<FRVT::Image> &faces, FRVT::TemplateRole, std::vector<uint8_t> &templ,
                                  std::vector<FRVT::EyePair> &eyeCoordinates) {
        templ.clear(); eyeCoordinates.clear();
        double best = -INFINITY;
        for (const FRVT::Image &im : faces) {                 // several photos of one person: keep the one whose eyes the
            std::vector<uint8_t> t; fl2::Eyes e; fl2::Box b;  // eye law is surest of
            int r = (im.data && im.width && im.height && (im.depth == 24 || im.depth == 8))
                        ? fl2::make_template(im.data.get(), im.width, im.height, im.depth / 8, t, e, b) : -1;
            eyeCoordinates.push_back(eye_pair(e, im));
            if (r == 0 && e.ev > best) { best = e.ev; templ = t; }
        }
        if (templ.empty()) {
            fl2::failed_template(templ);
            return FRVT::ReturnStatus(FRVT::ReturnCode::FaceDetectionError, "no face with two eyes found");
        }
        return FRVT::ReturnStatus(FRVT::ReturnCode::Success);
    }

    FRVT::ReturnStatus createIrisTemplate(const std::vector<FRVT::Image> &, FRVT::TemplateRole, std::vector<uint8_t> &,
                                          std::vector<FRVT::IrisAnnulus> &) override {
        return FRVT::ReturnStatus(FRVT::ReturnCode::NotImplemented);
    }

    FRVT::ReturnStatus createFaceTemplate(const FRVT::Image &image, FRVT::TemplateRole role, std::vector<std::vector<uint8_t>> &templs,
                                          std::vector<FRVT::EyePair> &eyeCoordinates) override {
        try { return everyone(image, role, templs, eyeCoordinates); }
        catch (const std::bad_alloc &) { templs.assign(1, std::vector<uint8_t>()); fl2::failed_template(templs[0]); eyeCoordinates.assign(1, FRVT::EyePair());
                                         return FRVT::ReturnStatus(FRVT::ReturnCode::MemoryError); }
        catch (...) { templs.assign(1, std::vector<uint8_t>()); fl2::failed_template(templs[0]); eyeCoordinates.assign(1, FRVT::EyePair());
                      return FRVT::ReturnStatus(FRVT::ReturnCode::UnknownError); }
    }

    FRVT::ReturnStatus everyone(const FRVT::Image &image, FRVT::TemplateRole, std::vector<std::vector<uint8_t>> &templs,
                                std::vector<FRVT::EyePair> &eyeCoordinates) {
        templs.clear(); eyeCoordinates.clear();
        std::vector<fl2::Eyes> eyes;
        int r = (image.data && image.width && image.height && (image.depth == 24 || image.depth == 8))
                    ? fl2::make_templates_all(image.data.get(), image.width, image.height, image.depth / 8, templs, eyes) : -1;
        if (r != 0 || templs.empty()) {
            templs.assign(1, std::vector<uint8_t>());
            fl2::failed_template(templs[0]);
            eyeCoordinates.assign(1, FRVT::EyePair());
            return FRVT::ReturnStatus(FRVT::ReturnCode::FaceDetectionError, "no face with two eyes found");
        }
        for (const fl2::Eyes &e : eyes) eyeCoordinates.push_back(eye_pair(e, image));
        return FRVT::ReturnStatus(FRVT::ReturnCode::Success);
    }

    FRVT::ReturnStatus matchTemplates(const std::vector<uint8_t> &verifTemplate, const std::vector<uint8_t> &enrollTemplate,
                                      double &score) override {
        try {
            bool ok = false;
            double s = fl2::match_templates(verifTemplate, enrollTemplate, true, ok);
            if (!ok) { score = -1.0; return FRVT::ReturnStatus(FRVT::ReturnCode::VerifTemplateError); }
            score = s + 127.0;
            return FRVT::ReturnStatus(FRVT::ReturnCode::Success);
        } catch (...) { score = -1.0; return FRVT::ReturnStatus(FRVT::ReturnCode::MatchError); }
    }

private:
    static void failed(std::vector<uint8_t> &templ, std::vector<FRVT::EyePair> &eyes, size_t n) {
        fl2::failed_template(templ);
        eyes.assign(n, FRVT::EyePair());
    }

    static FRVT::EyePair eye_pair(const fl2::Eyes &e, const FRVT::Image &im) {
        // the SUBJECT's left eye is on the image's right
        auto in = [&](double x, double y) { return e.ok && x >= 0 && y >= 0 && x < im.width && y < im.height; };
        bool l = in(e.rx, e.ry), r = in(e.lx, e.ly);
        return FRVT::EyePair(l, r, l ? (uint16_t)std::lrint(e.rx) : 0, l ? (uint16_t)std::lrint(e.ry) : 0,
                             r ? (uint16_t)std::lrint(e.lx) : 0, r ? (uint16_t)std::lrint(e.ly) : 0);
    }
};

}  // namespace

std::shared_ptr<FRVT_11::Interface> FRVT_11::Interface::getImplementation() { return std::make_shared<FaceLaw11>(); }
